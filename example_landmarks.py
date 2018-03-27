import pyVSR
from pyVSR import tcdtimit
from os import path


def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_dependent',
        remove_sa=True
    )

    experiment = pyVSR.AVSR(num_threads=4)

    file_dict = dict()
    output_dir = './run/features/landmarks/'
    for file in (train+test):
        file_dict[file] = path.join(
            output_dir,
            path.splitext(file.split('tcdtimit/')[-1])[0] + '.csv')

    experiment.extract_save_features(
        files=file_dict,
        feature_type='landmarks',
        extract_opts=None,
    )


if __name__ == "__main__":
    main()
