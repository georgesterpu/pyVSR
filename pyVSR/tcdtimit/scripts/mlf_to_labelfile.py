from pyVSR.pyVSR.tcdtimit.files import request_files
from os import path
from itertools import compress

def main():
    mlf = '/run/media/john_tukey/work/phd/32.seq2seq_feature/myseq2seq/pyVSR/pyVSR/tcdtimit/htkconfigs/allVisemes.mlf'

    train, test = request_files(dataset_dir='',
                                protocol='speaker_dependent',
                                remove_sa=False)

    train = [path.splitext(file)[0] for file in train]
    test = [path.splitext(file)[0] for file in test]

    with open(mlf, 'r') as f:
        contents = f.read().splitlines()

    labels_dict = dict()

    for file in train+test:
        print(file)
        transcript = _get_transcript_from_mlf(contents, file)
        labels_dict[file] = transcript

    with open('viseme_labels', 'w') as f:
        for k,v in labels_dict.items():
            f.write(k + ' ' + v + '\n')



def _get_transcript_from_mlf(mlf, sentence_id):
    feature_name = sentence_id.replace('/', '_') + '.lab'
    start = [line_nr for line_nr, line in enumerate(mlf) if feature_name in line]
    # if len(start) != 1:
    #     print(start)
    #     raise Exception('')
    start = start[0]

    count = 1
    transcript = ''
    while True:
        label = mlf[start+count].split(' ')
        if len(label) == 1:
            break
        transcript += label[-1]
        count += 1

    return transcript

if __name__ == '__main__':
    main()