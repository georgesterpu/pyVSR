from os import path
import pathlib2 as path2
from natsort import natsorted
from sys import argv
import pprint
import re

_current_path = path.abspath(path.dirname(__file__))

split = re.compile("_|-") 

def request_files(dataset_dir,
                  protocol='speaker_independent',
                  speaker_id=None, content="video", condition="none"):

    files = get_files(dataset_dir, content, condition)
    speakers = get_speakers(files)

    if protocol == 'speaker_dependent':
        train, dev, test = _preload_files_speaker_dependent(files, speaker_id, utterance_types)
    elif protocol == 'speaker_independent':
        train, dev, test = _preload_files_speaker_independent(files, speakers, content, condition)
    else:
        raise Exception('undefined dataset split protocol')

    return natsorted(train), natsorted(dev), natsorted(test),


def get_files(dataset_dir, content="video", condition=None):

    p = path2.Path(dataset_dir)

    if content == "video":
        p = p.joinpath("Lips")
        files = p.glob("*.mat")
    elif content == "audio":
        conditions = p.list
        p = p.joinpath("Audio").joinpath(condition)
        if p.exists() and p.is_dir():
            files = p.glob("*.mfcc")
        else:
            raise Exception("unknown condition: " + condition)
    else:
        raise Exception("unknown content: " + content)
    
    #the glob returns a generator that is empty once used
    return [f for f in files]


def get_speakers(files):
    
    return list(set([get_speaker(f) for f in files]))

def get_speaker(file_):
    return split.split(file_.stem)[1]


#speaker_dependent means: we have some speakers that we trained on in the dev/test sets
def _preload_files_speaker_dependent(files, speaker_id):

    raise Exception("speaker dependent protocol not implemented")

    ### NEED TO BE CREATIVE HERE
    ## we can basically split along repetitions but it's very little data

    #60/20/20 split by recursive split
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(files, test_size=0.20, random_state=0)
    train, dev = train_test_split(train, test_size=0.25, random_state=0)

    return train, dev, test

def _preload_files_speaker_independent(files, speakers, content="video", condition=None):

    #60/20/20 split by recursive split over speakers
    from sklearn.model_selection import train_test_split
    strain, stest = train_test_split(speakers, test_size=0.20, random_state=0)
    strain, sdev = train_test_split(strain, test_size=0.25, random_state=0)

    #map all files to their speaker
    speaker_files = {}
    for file_ in files:                                                    
        speaker = get_speaker(file_)
        try:
            speaker_files[speaker].append(file_)
        except:
            speaker_files[speaker] = [file_]

    train_files = []
    dev_files = []
    test_files = []
    #for each subset
    for sset, fset in [(strain, train_files),(sdev, dev_files),(stest, test_files)]:
        for speaker in sset:
            fset.extend(speaker_files[speaker])

    return train_files, dev_files, test_files

if __name__ == "__main__":

    print argv[0],": ",argv[1]
    pp = pprint.PrettyPrinter(indent=4)

    train, dev, test = request_files(argv[1], protocol='speaker_independent',
                  speaker_id=None, content="video", condition="none")

    print "train set:"
    pp.pprint(train)
    print "dev set:"
    pp.pprint(dev)
    print "test set:"
    pp.pprint(test)

