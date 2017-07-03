import pyVSR
from pyVSR import tcdtimit

def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='single_volunteer',
        speaker_id='34M')

    train_feat = pyVSR.utils.files_to_features(train, extension='.htk')
    test_feat = pyVSR.utils.files_to_features(test, extension='.htk')

    pyVSR.run(
        train_files=train_feat,
        test_files=test_feat,
        feature_dir='./run/features/htk_dct/DCT/',
        hmm_states=3,
        mixtures=(2, 3, 5, 7, 9, 11, 14, 17, 20),
        language_model=False,
        config_dir='./pyVSR/tcdtimit/htkconfigs/',
        report_results=('train', 'test'),
        experiment_name='dct_34M'
    )


if __name__ == '__main__':
    main()
