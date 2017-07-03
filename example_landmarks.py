import sys
import pyVSR
from pyVSR import tcdtimit


def main(argv):
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='single_volunteer',
        speaker_id='34M'
    )

    experiment = pyVSR.AVSR(num_threads=4)

    experiment.extract_save_features(
        files=train + test,
        feature_type='landmarks',
        extract_opts=None,
        output_dir='./run/features/facial_landmarks/'
    )

if __name__ == "__main__":
    main(sys.argv)
