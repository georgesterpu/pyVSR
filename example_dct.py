import pyVSR

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/ouluvs2/orig'
    train, test = ouluvs2.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_independent')

    experiment = pyVSR.AVSR(num_threads=4)

    experiment.extract_save_features(
        files=train + test,
        feature_type='roi',
        extract_opts={
            'align': True,
            'gpu': True,
            'color': True,
            'border': 15,
            'window_size': (36, 36),
            'num_subdirs_to_file' : 1,
        },
        output_dir='./run/features/roi_ouluvs2no/'
    )

if __name__ == "__main__":
    main()
