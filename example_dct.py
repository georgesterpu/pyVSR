import pyVSR
from os import path
from pyVSR import tcdtimit
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_dependent',
        remove_sa=True)

    file_dict = dict()
    output_dir = './run/features/roi/'
    for file in (train+test):
        file_dict[file] = path.join(
            output_dir,
            path.splitext(file.split('tcdtimit/')[-1])[0] + '.h5')

    experiment = pyVSR.AVSR(num_threads=4)

    experiment.extract_save_features(
        files=file_dict,
        feature_type='roi',
        extract_opts={
            'align': True,
            'gpu': True,
            'color': True,
            'border': 15,
            'window_size': (36, 36),
        },
    )

    feat_dict = dict()
    outdir = './run/features/dct/'
    for file in file_dict.values():
        feat_dict[file] = path.join(
            outdir,
            path.splitext(file.split('roi/')[-1])[0] + '.htk'
        )

    experiment.process_features_write_htk(
        files=feat_dict,
        feature_type='dct',
        process_opts={
            'compute_dct': True,
            'mask': '1-44',
            'keep_basis': True,
            'delta': True,
            'double_delta': True,
            'derivative_order': 'first',
            'interpolation_factor': 1,
        },
        frame_rate=30,
    )


if __name__ == "__main__":
    main()
