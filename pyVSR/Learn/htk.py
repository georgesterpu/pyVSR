from os import path, makedirs, remove, listdir
from subprocess import list2cmdline, run, Popen, PIPE
import numpy as np
from ..utils import read_htk_header
from ..tcdtimit.files import phoneme_file, phoneme_list, viseme_file, viseme_list


class HTKSys(object):
    r"""
    AVSR Training and Recognition using HTK
    """

    def __init__(self,
                 train_files,
                 test_files,
                 htk_dir,
                 hmm_states=3,
                 mixtures=(1,),
                 language_model=False,
                 config_dir=None,
                 report_results=('test',),
                 num_threads=8):
        r"""
        Class constructor
        :param train_files: Train files split
        :param test_files: Test files split
        :param htk_dir: Directory containing the .htk feature files
        :param hmm_states: Number of states per HMM
        :param mixtures: Tuple of increasing values representing the number of GMM mixtures
        :param language_model: Use a language model (True/False)
        :param report_results: Tuple containing the dataset splits (train, test) 
        """

        # argument copy
        self._trainfeat = train_files
        self._testfeat = test_files
        self._feature_path = htk_dir
        self._hmmstates = hmm_states
        self._mixtures = mixtures
        self._languageModel = language_model
        self._report_results = report_results

        # static dataset configs
        self._config = path.join(config_dir, 'settings.conf')
        self._viseme_list = path.join(config_dir, 'viseme_list')
        self._viseme_dict = path.join(config_dir, 'viseme_dict')
        self._grammar = path.join(config_dir, 'grammar')
        self._labels = path.join(config_dir, 'allVisemes.mlf')
        self._HViteConf = path.join(config_dir, 'HVite.conf')

        # dynamically generated
        self._run_dir = './run/'
        self._hmm_dir = './run/hmms/'
        self._cleanup_run_dir()
        self._hmm_proto = './run/proto'
        self._word_net = None  # generated at runtime
        self._gen_wordnet('./run/wdnet')

        # library initialisation
        self._num_runs = 0
        self._num_threads = num_threads
        self._feature_size = self._get_feature_size()

    def _cleanup_run_dir(self):
        try:
            allfiles = [file for file in listdir(self._run_dir) if path.isfile(self._run_dir + file)]
            for f in allfiles:
                remove(self._run_dir + f)
            if path.isdir(self._hmm_dir):
                from shutil import rmtree
                rmtree(self._hmm_dir)
        except Exception:
            print('cleaning failed')

    def _get_feature_size(self):
        htkfile = self._feature_path + path.splitext(self._trainfeat[0])[0] + '.htk'
        header = read_htk_header(htkfile)  # read header of the first file
        return header[2]//4  # sampSize is in bytes

    def run(self):
        r"""
        HTK-Based training
        1. Run HCompV to initialize a prototype HMM with global means and variances
        2. Replicate the prototype for each viseme
        3. Apply multiple iterations of HRest for Baum-Welch embedded training
        :return:
        """
        self._initialize_stats()
        self._replicate_proto()
        self._embedded_reestimation(num_times=5)
        self._fix_silence_viseme()
        self._embedded_reestimation(num_times=5)
        for case in self._report_results:
            if case == 'test':
                self.test(self._testfeat)
                self.print_results(1, case)
            elif case == 'train':
                self.test(self._trainfeat)
                self.print_results(1, case)

        # for i in range(2,self._finalMixtures+1):
        for i in self._mixtures:
            self._increase_mixtures(i)
            self._embedded_reestimation(num_times=5)

            for case in self._report_results:
                if case == 'test':
                    self.test(self._testfeat)
                    self.print_results(i, case)
                elif case == 'train':
                    self.test(self._trainfeat)
                    self.print_results(i, case)

    def test(self, features):
        """
        HTK-Based decoding
        Runs HVite on the feature list
        :param features:
        :return:
        """
        self._decode(features)

    def _decode(self, features):
        currdir = self._hmm_dir + 'hmm' + str(self._num_runs) + '/'
        outfile = './run/predicted.mlf'
        self.predicted_labels = outfile
        scp = './run/test.scp'
        self._gen_feature_scp('./run/test.scp', features)

        scp_list = _split_scp(scp, num_threads=self._num_threads)
        threadlist = []

        for th in range(self._num_threads):
            # cmd = ['HVite', '-C', self._HViteConf, '-H', currdir + 'vFloors', '-H', currdir + 'hmmdefs', '-S',
            #        scp_list[th], '-l', '\'*\'', '-i', outfile+'.part' +str(th) , '-w', self._word_net,
            #        '-p', '0.0', '-s', '1.0', self._viseme_dict, self._viseme_list]

            cmd = ['HVite', '-C', self._HViteConf, '-H', currdir + 'vFloors', '-H', currdir + 'hmmdefs', '-S',
                   scp_list[th], '-l', '\'*\'', '-i', outfile + '.part' + str(th), '-w', self._word_net,
                   self._viseme_dict, self._viseme_list]

            print(list2cmdline(cmd))
            proc = Popen(cmd)
            threadlist.append(proc)

        # sync threads
        for proc in threadlist:
            proc.wait()

        # merge parts, discarding the header, do cleanup
        if path.isfile(outfile):
            remove(outfile)
        with open(outfile, 'a') as of:
            of.write('#!MLF!#\n')  # initial header

            for part in range(self._num_threads):
                partfile = outfile+'.part' + str(part)
                with open(partfile, 'r') as f:
                    contents = f.readlines()
                of.writelines(contents[1:])  # ignore the header #!MLF!#
                remove(partfile)

        for file in scp_list:
            cmd = ['rm ' + file]
            run(cmd, shell=True, check=True)

    def _initialize_stats(self):
        firstdir = './run/hmms/hmm' + str(self._num_runs) + '/'
        makedirs(firstdir, exist_ok=True)

        self._gen_proto(vecsize=self._feature_size, nstates=self._hmmstates)
        scp = './run/train.scp'
        self.trainscp = scp
        self._gen_feature_scp('./run/train.scp', self._trainfeat)

        cmd = ['HCompV', '-C', self._config, '-f', '0.01', '-m', '-S', self.trainscp, '-M', firstdir, self._hmm_proto]
        print(list2cmdline(cmd))
        run(cmd, check=True)

    def _increase_mixtures(self, nmix):
        scp = self._gen_edit_script_num_mixtures(nmix)

        prevdir = self._hmm_dir + 'hmm' + str(self._num_runs) + '/'
        nextdir = self._hmm_dir + 'hmm' + str(self._num_runs + 1) + '/'
        self._num_runs += 1
        makedirs(nextdir, exist_ok=True)

        cmd = ['HHEd', '-H', prevdir + 'vFloors', '-H', prevdir + 'hmmdefs', '-M', nextdir, scp,
               self._viseme_list]
        print(list2cmdline(cmd))
        run(cmd, check=True)

    def _fix_silence_viseme(self):
        edit_script = self._gen_edit_script_silence_vis()

        self._num_runs += 1

        prevdir = self._hmm_dir + 'hmm' + str(self._num_runs - 1) + '/'
        nextdir = self._hmm_dir + 'hmm' + str(self._num_runs) + '/'
        makedirs(nextdir, exist_ok=True)

        cmd = ['HHEd', '-H', prevdir + 'vFloors', '-H', prevdir + 'hmmdefs', '-M',
               nextdir, edit_script, self._viseme_list]

        print(list2cmdline(cmd))
        run(cmd, check=True)

    def _gen_edit_script_silence_vis(self):
        fname = './run/sil.hed'
        with open(fname, 'w') as f:
            for s in range(self._hmmstates+1, 2, -1):
                for j in range(s-1, 1, -1):
                    f.write('AT ' + str(s) + ' ' + str(j) + ' 0.2 {S.transP}\n')

            f.write('AT 2 ' + str(self._hmmstates + 1) + ' 0.2 {S.transP}\n')
        return fname

    def _gen_edit_script_num_mixtures(self, num_mixtures):
        file_name = './run/mix_' + str(num_mixtures) + '.hed'
        with open(file_name, 'w') as f:
            f.write('MU ' + str(num_mixtures) + ' {*.state[2-' + str(self._hmmstates + 1) + '].mix}\n')
        return file_name

    def _gen_proto(self, vecsize, nstates):
        lines = ['~o <VecSize> ' + str(vecsize) + ' <USER>\n',
                 '~h "proto"\n',
                 '<BeginHMM>\n',
                 '<NumStates> ' + str(nstates + 2) + '\n']

        for s in range(2, nstates+2):
            lines.append('<State> ' + str(s) + '\n')
            lines.append('<Mean> ' + str(vecsize) + '\n')
            lines.append('0.0 '*vecsize + '\n')
            lines.append('<Variance> ' + str(vecsize) + '\n')
            lines.append('1.0 ' * vecsize + '\n')

        lines.append('<TransP> ' + str(nstates+2) + '\n')
        lines.append('0.0 1.0 ' + '0.0 '*nstates + '\n')

        for i in range(1, nstates+1):
            lines.append('0.0 ' * i + '0.5 0.5 ' + '0.0 ' * (nstates-i) + '\n')

        lines.append('0.0 ' * (nstates+2) + '\n')

        lines.append('<EndHMM>\n')

        with open(self._hmm_proto, 'w') as f:
            f.writelines(lines)

    def _gen_feature_scp(self, scpname, features):
        lines = []
        for file in features:
            line = self._feature_path + path.splitext(file)[0] + '.htk\n'
            lines.append(line)

        with open(scpname, 'w') as f:
            f.writelines(lines)

    def _gen_wordnet(self, wdnet):

        if self._languageModel is True:
            new_hmmlist = append_to_file(self._viseme_list, ('!ENTER', '!EXIT'))
            self._viseme_dict = append_to_file(self._viseme_dict, ('!ENTER []', '!EXIT []'))

            cmd = ['HLStats -b ./run/bigrams -o ' + self._viseme_list + ' ' + self._labels]
            print(list2cmdline(cmd))
            run(cmd, check=True, shell=True)

            cmd = ['HBuild -n ./run/bigrams ' + new_hmmlist + ' ' + wdnet]
            print(list2cmdline(cmd))
            run(cmd, check=True, shell=True)
            self._word_net = wdnet

        else:
            cmd = ['HParse', self._grammar, wdnet]
            print(list2cmdline(cmd))
            run(cmd, check=True)
            self._word_net = wdnet

    def _replicate_proto(self):

        # copied this function from a tutorial
        # TODO - refactor this function, rootdirs as arguments

        hmm0_dir = path.join(self._hmm_dir, 'hmm0')

        # read proto lines
        hmm_proto = path.join(hmm0_dir, 'proto')
        f = open(hmm_proto, 'r')
        proto_lines = []
        for l in f:
            l = l.rstrip('\r\n')
            proto_lines.append(l)
        f.close()

        # read vfloor lines
        vfloor_file = path.join(hmm0_dir, 'vFloors')
        v = open(vfloor_file, 'r')
        vfloor_lines = []
        for l in v:
            l = l.rstrip('\r\n')
            vfloor_lines.append(l)
        v.close()

        # append first lines of hmm proto to vfloors file
        v = open(vfloor_file, 'w')
        for l in proto_lines[0:3]:
            v.write('%s\n' % l)
        for l in vfloor_lines:
            v.write('%s\n' % l)
        v.close()

        # read phoneme list
        pl = open(self._viseme_list, 'r')
        phones = []
        for p in pl:
            p = p.rstrip('\r\n')
            phones.append(p)
        pl.close()

        # for each phone copy prototype
        hmmdefs = path.join(hmm0_dir, 'hmmdefs')
        h = open(hmmdefs, 'w')
        for p in phones:
            h.write('~h \"%s\"\n' % p)
            for pl in proto_lines[4:]:
                h.write('%s\n' % pl)
        h.close()

    def _embedded_reestimation(self, num_times, binary=False, pruning='off', stats=False, num_threads=8):
        """
        :param num_times:
        :return:
        """

        if binary is False:
            bincfg = []
        elif binary is True:
            bincfg = ['-B']
        else:
            raise Exception('error estting parameters')

        if pruning == 'off':
            prune_cfg = []
        elif pruning == 'on':
            prune_cfg = ['-t', '250.0', '150.0', '1000.0']
        else:
            raise Exception('error estting parameters')

        if stats is False:
            statscfg = []
        elif stats is True:
            statscfg = ['-s', self._hmm_dir + 'stats']
        else:
            raise Exception('error estting parameters')

        scp_list = _split_scp(self.trainscp, num_threads=num_threads)

        for loop in range(num_times):

            self._num_runs += 1

            previous_dir = self._hmm_dir + 'hmm' + str(self._num_runs - 1) + '/'
            current_dir = self._hmm_dir + 'hmm' + str(self._num_runs) + '/'
            makedirs(current_dir, exist_ok=True)

            threadlist = []

            for thread in range(len(scp_list)):

                cmd = ['HERest'] + bincfg + ['-C', self._config, '-I', self._labels] + prune_cfg + statscfg + \
                      ['-S', scp_list[thread], '-H', previous_dir + 'vFloors', '-H', previous_dir + 'hmmdefs',
                       '-M', current_dir, '-p', str(thread+1), self._viseme_list]

                print(list2cmdline(cmd))
                proc = Popen(cmd)
                threadlist.append(proc)

            # sync threads
            for proc in threadlist:
                proc.wait()

            # Run final HERest to collect the accummulators

            cmd = ['HERest'] + bincfg + ['-C', self._config, '-I', self._labels] + prune_cfg + statscfg + \
                  ['-H', previous_dir + 'vFloors', '-H', previous_dir + 'hmmdefs',
                   '-M', current_dir, '-p', '0', self._viseme_list, current_dir + '*.acc']

            run(list2cmdline(cmd), shell=True, check=True)

            # cleanup folder (remove accs, scp.i)
            cmd = ['rm ' + current_dir + '*.acc']
            run(cmd, shell=True, check=True)

        for file in scp_list:
            cmd = ['rm ' + file]
            run(cmd, shell=True, check=True)

    def print_results(self, nmix, case):

        cmd = ['HResults', '-I', self._labels, '-f', '-p', self._viseme_list, self.predicted_labels]
        print(list2cmdline(cmd))
        with open('./run/results_' + case + '_' + str(nmix)+'_mixtures.txt', 'w') as logfile:
            run(cmd, check=True, stdout=logfile)


# r"""these functions are not part of the class"""

def _split_scp(scp, num_threads):
    with open(scp, 'r') as fr:
        contents = fr.readlines()

    num_lines = len(contents)
    avg = int(num_lines//num_threads)
    idx_start = np.arange(num_threads) * avg
    idx_end = np.arange(1, num_threads + 1) * avg
    idx_end[-1] = num_lines

    scplist = []
    for i in range(num_threads):
        partial_scp = scp + '.part' + str(i)
        with open(partial_scp, 'w') as fw:
            fw.writelines(contents[idx_start[i]:idx_end[i]])
        scplist.append(partial_scp)

    return scplist


def read_result_file(file):
    r"""
    Reads the contents of a HTK result log file
    :param file: result log
    :return: correctness, accuracy
    """
    with open(file, 'r') as f:
        contents = f.read().splitlines()

    idx_acc = contents[6].index('Acc=')
    acc = float(contents[6][idx_acc+4:idx_acc+9])

    idx_corr = contents[6].index('Corr=')
    corr = float(contents[6][idx_corr + 5:idx_corr + 10])

    return corr, acc


def read_result_str(result_string):
    idx_acc = result_string.index('Acc=')
    acc = float(result_string[idx_acc + 4:idx_acc + 9].strip(','))
    idx_corr = result_string.index('Corr=')
    corr = float(result_string[idx_corr + 5:idx_corr + 10].strip(','))
    return corr, acc


def append_to_file(hmmlist, items):
    with open(hmmlist, 'r') as f:
        contents = f.read().splitlines()

    oldfile = path.split(hmmlist)[1]
    newfile = './run/' + oldfile + '_lm'

    for item in items:
        contents.append(item)

    with open(newfile, 'w') as f:
        f.writelines(line+'\n' for line in contents)

    return newfile


def compute_results2(predicted_labels, ground_truth_file, unit_list_file):

    cmd = ['HResults', '-I', ground_truth_file, unit_list_file, predicted_labels]

    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    results = p.communicate()[0]
    return read_result_str(results.decode('utf-8'))


def compute_results3(predicted_labels, unit):

    if unit == 'viseme':
        return compute_results2(
            predicted_labels,
            viseme_file,
            viseme_list)
    elif unit == 'phoneme':
        return compute_results2(
            predicted_labels,
            phoneme_file,
            phoneme_list
        )
    else:
        raise Exception('unknown unit: {}'.format(unit))