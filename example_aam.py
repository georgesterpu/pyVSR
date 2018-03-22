import pyVSR
from pyVSR import tcdtimit
from os import path


def main():
    dataset_dir = '/run/media/john_tukey/download/datasets/adapt/tcdtimit/'
    train, test = tcdtimit.files.request_files(
        dataset_dir=dataset_dir,
        protocol='speaker_dependant',
        remove_sa=True)

    experiment = pyVSR.AVSR(num_threads=4)

    file_lms_dict = dict()
    lms_dir = './run/features/landmarks/'
    for file in train[::14]:
        file_lms_dict[file] = path.join(
            lms_dir,
            path.splitext(file.split('tcdtimit/')[-1])[0])

    experiment.extract_save_features(
        files=file_lms_dict,  # only a few training files
        feature_type='aam',
        extract_opts={
           'warp': 'holistic',
           'resolution_scales': (0.25, 0.5, 1.0),
           'patch_shape': ((5, 5), (10, 10), (17, 17)),
           'max_shape_components': 20,
           'max_appearance_components': 150,
           'diagonal': 150,
           'features': 'no_op',
           'landmark_group': 'pts_face',
           'confidence_thresh': 0.94,
           'kept_frames': 0.03,
           'greyscale': False,
           'model_name': 'face_hnop_34M.pkl'},
        output_dir='./run/features/aam_model/'
    )

    htk_dict = dict()
    out_dir = './run/features/aam_htk/'
    for file in train+test:
        htk_dict[file] = path.join(out_dir, path.splitext(file.split('tcdtimit/')[-1])[0] + '.htk')

    experiment.process_features_write_htk(
        files=htk_dict,
        feature_type='aam',
        process_opts={
            'face_detector': 'dlib',
            'landmark_fitter': 'aam',
            'aam_fitter': './run/features/aam_model/face_hnop_34M.pkl',
            'parameters_from': 'lk_fitting',
            'projection_aam': './run/features/aam_model/face_hnop_34M.pkl',
            'shape': 'face',
            'part_aam': None,
            'confidence_thresh': 0.84,
            'shape_components': [10, 15, 20],
            'appearance_components': [20, 30, 150],
            'max_iters': [10, 10, 5],
            'log_errors': False,
            'log_dir': '34M/log_demo'},
    )


if __name__ == "__main__":
    main()
