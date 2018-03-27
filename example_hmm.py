import pyVSR
from pyVSR import tcdtimit
from os import path

def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_dependent',
        remove_sa=True)

    featdir = './run/features/dct/'

    pyVSR.run(
        train_files=tcdtimit_feature_dict(train, featdir),
        test_files=tcdtimit_feature_dict(test, featdir),
        hmm_states=3,
        mixtures=(2, 3, 5, 7, 9, 11, 14, 17, 20),
        language_model=False,
        config_dir='./pyVSR/tcdtimit/htkconfigs/',
        report_results=('train', 'test'),
        experiment_name='dct_volunteers',
        num_threads=4,
    )


def tcdtimit_feature_dict(files, feat_dir):
    feature_dict = dict()
    for file in files:
        htk_file = path.join(
            feat_dir,
            path.splitext(file.split('tcdtimit/')[-1])[0] + '.htk')
        sentence_id = path.splitext(file.split('tcdtimit/')[-1])[0]
        feature_dict[sentence_id] = htk_file
    return feature_dict


if __name__ == '__main__':
    main()
