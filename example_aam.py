import sys
import pyVSR
from pyVSR import tcdtimit


def main(argv):
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='single_volunteer',
        speaker_id='34M')

    experiment = pyVSR.AVSR(num_threads=4)

    experiment.extract_save_features(
        files=train[::14],  # only a few training files
        feature_type='aam',
        extract_opts={
           'warp':'holistic',
           'resolution_scales': (0.25, 0.5, 1.0),
           'patch_shape':((5,5), (10,10), (17,17)),
           'max_shape_components':20,
           'max_appearance_components': 150,
           'diagonal': 150,
           'features': 'no_op',
           'landmark_dir': './run/features/facial_landmarks/',
           'landmark_group': 'pts_face',
           'confidence_thresh':0.94,
           'kept_frames': 0.03,
           'greyscale':False,
           'model_name': 'face_hnop_34M.pkl'},
        output_dir='./run/features/aam/'
    )

    experiment.process_features_write_htk(
        files=test,
        feature_type='aam',
        process_opts={
            'face_detector': 'dlib',
            'landmark_fitter': 'ert',
            'aam_fitter': './run/features/aam/face_hnop_34M.pkl',
            'parameters_from': 'aam_projection',
            'projection_aam': './run/features/aam/face_hnop_34M.pkl',
            'shape': 'face',
            'part_aam': None,
            'confidence_thresh': 0.84,
            'shape_components': [10, 15, 20],
            'appearance_components': [20, 30, 150],
            'max_iters': [10, 10, 5],
            'landmark_dir': './run/features/facial_landmarks/',
            'log_errors': False,
            'log_dir': '34M/log_demo'},
        output_dir='./run/features/htk_aam_demo/'
    )


if __name__ == "__main__":
    main(sys.argv)
