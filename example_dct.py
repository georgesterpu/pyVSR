import pyVSR
from pyVSR import tcdtimit
from pyVSR.utils import files_to_features


def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='single_volunteer',
        speaker_id='34M')

    experiment = pyVSR.AVSR(num_threads=8)

    experiment.extract_save_features(
        files=train+test,
        feature_type='dct',
        extract_opts={
            'roi_extraction': 'dlib',
            'boundary_proportion': 0.7,
            # 'roi_dir': './run/features/rois/',
            'window_size': (36, 36)
        },
        output_dir='./run/features/dct/'
    )

    dct_train = files_to_features(train, extension='.h5')
    dct_test = files_to_features(test, extension='.h5')

    experiment.process_features_write_htk(
        files=dct_train + dct_test,
        feature_dir='./run/features/dct/',
        feature_type='dct',
        process_opts={
            'mask': '1-44',
            'keep_basis': True,
            'delta': True,
            'double_delta': True,
            'derivative_order': 'fourth',
            'interpolation_factor': 1},
        frame_rate=30,
        output_dir='./run/features/htk_dct/'
    )


if __name__ == "__main__":
    main()
