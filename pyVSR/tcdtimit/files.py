from os import path, makedirs
from glob import glob
from natsort import natsorted

_current_path = path.abspath(path.dirname(__file__))
viseme_file = path.join(_current_path, './htkconfigs/allVisemes.mlf')
phoneme_file = path.join(_current_path, './htkconfigs/allPhonemes.mlf')
viseme_list = path.join(_current_path, './htkconfigs/viseme_list')
phoneme_list = path.join(_current_path, './htkconfigs/phoneme_list')

volunteers = ('01M', '02M', '03F', '04M', '05F', '06M', '07F', '08F', '09F', '10M',
              '11F', '12M', '13F', '14M', '15F', '16M', '17F', '18M', '19M', '20M',
              '21M', '22M', '23M', '24M', '25M', '26M',        '28M', '29M', '30F',
              '31F', '32F', '33F', '34M',        '36F', '37F', '38F', '39M', '40F',
              '41M', '42M', '43F', '44F', '45F', '46F', '47M', '48M', '49F', '50F',
              '51F', '52M',        '54M', '55F', '56M', '57M', '58F', '59F',)
# volunteers 27, 35, 53 excluded for their non-irish accent (e.g. spanish, british)


def request_files(dataset_dir, protocol=None, speaker_type=None, gender=None, speaker_id=None, remove_sa=False):
    r"""Generates the train/test split according to predefined protocols.
    If no protocol is defined, the function attempts to find all the video files located at `dataset_dir`
    and return a random train/test split.
    Parameters
    ----------
    dataset_dir
    protocol : `str` or `None`, optional
        Can be ``speaker_dependent``, ``speaker_independent``, ``single volunteer``
    speaker_type : 'str' or `None`, optional
        Can be ``volunteer`` or ``lipspeaker``
    gender : `str`, optional
        Can be ``females``, ``males`` or ``both``
    speaker_id : `str`, optional
        A three character string encoding the ID of a volunteer, .e.g. ``01M``
        
    Returns
    -------

    """
    if protocol == 'speaker_independent':
        train, test = _preload_files_speaker_independent(dataset_dir)
    elif protocol == 'speaker_dependent':
        train, test = _preload_files_speaker_dependent(dataset_dir)
    elif protocol == 'single_volunteer':
        train, test = _preload_files_single_volunteer(dataset_dir, speaker_id)
    else:
        files = _find_files(dataset_dir, speaker_type, gender)
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(files, test_size=0.30, random_state=0)

    if remove_sa is True:
        train = [file for file in train if 'sa1' not in file and 'sa2' not in file]
        test = [file for file in test if 'sa1' not in file and 'sa2' not in file]

    return natsorted(train), natsorted(test)


def _read_file_contents(file):
    with open(file, 'r') as ftr:
        contents = ftr.read().splitlines()
    return contents

def _preload_files_speaker_dependent(dataset_dir):
    r"""Speaker-dependent protocol
    Each speaker contributes with 67 training sentences
    and 31 testing sentences
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    """

    train_script = path.join(_current_path, 'splits/speaker-dependent/train.scp')
    test_script = path.join(_current_path, 'splits/speaker-dependent/test.scp')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script)]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script)]

    return train_files, test_files


def _preload_files_speaker_independent(dataset_dir):
    r"""Speaker-independent protocol
    There are 39 volunteers used for training
    and 17 for testing
    The remaining three are excluded
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    """

    train_script = path.join(_current_path, 'splits/speaker-independent/train.scp')
    test_script = path.join(_current_path, 'splits/speaker-independent/test.scp')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script)]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script)]

    return train_files, test_files


def _preload_files_single_volunteer(dataset_dir, speaker_id):
    r"""Loads the file of a single volunteer, maintaining the same train/test split
    from the speaker dependent protocol
    Parameters
    ----------
    dataset_dir
    speaker_id

    Returns
    -------

    """

    train_script = path.join(_current_path, 'splits/speaker-dependent/train.scp')
    test_script = path.join(_current_path, 'splits/speaker-dependent/test.scp')

    train_files = [path.join(dataset_dir, line) for line in _read_file_contents(train_script) if speaker_id in line]
    test_files = [path.join(dataset_dir, line) for line in _read_file_contents(test_script) if speaker_id in line]

    return train_files, test_files


def _find_files(dataset_dir=None, speaker_type='both', gender='both'):
    r"""Filters the files in the dataset by speaker type and gender
    Parameters
    ----------
    dataset_dir
    speaker_type
    gender

    Returns
    -------
    files

    """
    makedirs('./store/', exist_ok=True)
    contents, status = _read_flist_from_file(speaker_type, gender)

    if status is True:  # list already cached
        files = contents
    else:  # create list and cache it for future usage
        if gender == 'males':
            g_pattern = 'M'
        elif gender == 'females':
            g_pattern = 'F'
        elif gender == 'both':  # flame on
            g_pattern = '*'
        else:
            raise Exception('Invalid gender parameter')

        if speaker_type == 'volunteers':
            t_pattern = 'volunteers'
        elif speaker_type == 'lipspeakers':
            t_pattern = 'lipspeakers'
        elif speaker_type == 'both':
            t_pattern = '*'
        else:
            raise Exception('Invalid category parameter')

        pose = 'straightcam'
        if dataset_dir is None:
            raise Exception('Path to dataset unspecified')

        if dataset_dir[-1] != '/':
            dataset_dir += '/'

        files = glob(dataset_dir + t_pattern + '/*' + g_pattern + '/*/' + pose + '/*.mp4')

        files = [file for file in files
                 if file.find('27M') == -1
                 and file.find('35M') == -1
                 and file.find('53M') == -1]

        _write_flist_to_file(files, speaker_type, gender, pose)

    return files


def _read_flist_from_file(speaker_type, gender):
    import pickle
    files = None
    status = False
    pose = 'straightcam'

    fname = speaker_type + '_' + gender + '_' + pose + '.pkl'
    if path.exists('./store/' + fname):
        with open('./store/'+fname, 'rb') as f:
            files = pickle.load(f)
            status = True

    return files, status


def _write_flist_to_file(files, speaker_type, gender, pose):
    import pickle
    fname = speaker_type + '_' + gender + '_' + pose + '.pkl'
    if path. exists('./store/'+fname):
        pass
    else:
        with open('./store/'+fname, 'wb') as f:
            pickle.dump(files, f)


def read_sentence_labels(filename, unit='viseme'):
    r"""Finds the labels associated with a sentence
    in a .mlf label file
    Parameters
    ----------
    filename

    Returns
    -------
    label_seq : `list`
        A list of label symbols
    """
    file = path.splitext(path.split(filename)[1])[0]

    if unit == 'viseme':
        transcript = viseme_file
    elif unit == 'phoneme':
        transcript = phoneme_file
    else:
        raise Exception('only `viseme` and `phoneme` unit transcriptions are supported')

    with open(transcript, 'r') as f:
        contents = f.read()

    start = contents.find(file)
    end = contents.find('.\n', start)
    sentence_transcript = contents[start:end].splitlines()[1:]

    label_seq = [item.split()[-1] for item in sentence_transcript]
    return label_seq
