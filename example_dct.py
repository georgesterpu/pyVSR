import pyVSR
from pyVSR import tcdtimit
from pyVSR.utils import files_to_features

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_dependent')

    experiment = pyVSR.AVSR(num_threads=1)

    experiment.extract_save_features(
        files=train+test,
        feature_type='roi',
        extract_opts={
            'align': True,
            'gpu': True,
            'color': False,
            'border': 15,
            'window_size': (36, 36),
        },
        output_dir='./run/features/roi_gray/'
    )

    dct_train = files_to_features(train, extension='.h5')
    dct_test = files_to_features(test, extension='.h5')

    experiment.process_features_write_htk(
        files=dct_train + dct_test,
        feature_dir='./run/features/roi_gray/',
        feature_type='dct',
        process_opts={
            'compute_dct': True,
            'mask': '1-44',
            'keep_basis': True,
            'delta': True,
            'double_delta': True,
            'derivative_order': 'first',
            'interpolation_factor': 1},
        frame_rate=30,
        output_dir='./run/features/htk_dct_gray/'
    )


if __name__ == "__main__":
    main()
