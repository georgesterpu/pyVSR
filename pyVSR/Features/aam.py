from .feature import Feature
from .. import vsrmath
import menpo.io as mio
from ..utils import file_to_feature
from menpofit.aam import HolisticAAM, PatchAAM
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpo.feature import fast_dsift, dsift, hog, no_op
from menpo.base import LazyList
import numpy as np
from os import path
from .landmark import landmark_filter
from menpofit.dlib import DlibWrapper
from menpo.shape import PointCloud

current_path = path.abspath(path.dirname(__file__))


class AAMFeature(Feature):
    r"""
    Active Appearance Model (AAM) feature extraction
    """
    def __init__(self, files=None, feat_dir=None, extract_opts=None, process_opts=None, out_dir=None):

        self._allFiles = files
        self._featDir = feat_dir
        self._outDir = out_dir
        if extract_opts is not None:
            self._extractOpts = extract_opts

            self._warpType = extract_opts['warp']
            self._landmarkDir = extract_opts['landmark_dir']
            self._landmarkGroup = extract_opts['landmark_group']
            self._max_shape_components = extract_opts['max_shape_components']
            self._max_appearance_components = extract_opts['max_appearance_components']
            self._diagonal = extract_opts['diagonal']
            self._scales = extract_opts['resolution_scales']
            self._confidence_thresh = extract_opts['confidence_thresh']
            self._kept_frames = extract_opts['kept_frames']
            if extract_opts['features'] == 'fast_dsift':
                self._features = fast_dsift
            elif extract_opts['features'] == 'dsift':
                self._features = dsift
            elif extract_opts['features'] == 'hog':
                self._features = hog
            elif extract_opts['features'] == 'no_op':
                self._features = no_op
            else:
                raise Exception('Unknown feature type to extract, did you mean fast_dsift ?')

            if 'greyscale' in extract_opts.keys():
                self._greyscale = extract_opts['greyscale']
            else:
                self._greyscale = False

            self._outModelName = extract_opts['model_name']

        if process_opts is not None:
            # Face detection
            self._face_detect_method = process_opts['face_detector']
            if self._face_detect_method == 'dlib':
                from menpodetect import load_dlib_frontal_face_detector
                detector = load_dlib_frontal_face_detector()
            elif self._face_detect_method == 'opencv':
                from menpodetect import load_opencv_frontal_face_detector
                detector = load_opencv_frontal_face_detector()
            elif self._face_detect_method == 'dpm':
                from menpodetect.ffld2 import load_ffld2_frontal_face_detector
                detector = load_ffld2_frontal_face_detector()
            else:
                raise Exception('unknown detector, did you mean dlib/opencv/dpm?')

            self._face_detect = detector

            self._shape_components = process_opts['shape_components']
            self._appearance_components = process_opts['appearance_components']
            self._max_iters = process_opts['max_iters']

            self._fitter_type = process_opts['landmark_fitter']
            # Landmark fitter (pretrained ERT or AAM), actually loaded later to avoid pickling with Pool
            if self._fitter_type == 'aam':
                self._aam_fitter_file = process_opts['aam_fitter']

            # Parameters source
            # If fitting,
            self._parameters = process_opts['parameters_from']

            if self._parameters == 'aam_projection':
                self._projection_aam_file = process_opts['projection_aam']
                self._projection_aam = mio.import_pickle(self._projection_aam_file)
                self._projection_fitter = LucasKanadeAAMFitter(
                    aam=self._projection_aam,
                    lk_algorithm_cls=WibergInverseCompositional,
                    n_shape=self._shape_components,
                    n_appearance=self._appearance_components)
            else:
                pass

            self._confidence_thresh = process_opts['confidence_thresh']
            self._landmarkDir = process_opts['landmark_dir']

            self._shape = process_opts['shape']
            self._part_aam = process_opts['part_aam']

            self._log_errors = process_opts['log_errors']
            if self._log_errors is False:
                self._myresolver = None

            self._title = process_opts['log_title']

    def extract_save_features(self, files):

        # 1. fetch all video frames, attach landmarks
        frames = mio.import_video(files[0],
                                  landmark_resolver=self._myresolver,
                                  normalize=True,
                                  exact_frame_count=True)

        # frames = frames.map(AAMFeature._preprocess)
        idx_above_thresh, idx_lip_opening = landmark_filter(
            files[0],
            self._landmarkDir,
            threshold=self._confidence_thresh,
            keep=self._kept_frames)

        frames = frames[idx_above_thresh]
        frames = frames[idx_lip_opening]
        frames = frames.map(attach_semantic_landmarks)

        if self._greyscale is True:
            frames = frames.map(convert_to_grayscale)

        # initial AAM training
        if self._warpType == 'holistic':
            aam = HolisticAAM(frames,
                              group=self._landmarkGroup,
                              holistic_features=self._features,
                              reference_shape=None,
                              diagonal=self._diagonal,
                              scales=self._scales,
                              max_shape_components=self._max_shape_components,
                              max_appearance_components=self._max_appearance_components,
                              verbose=False)
        elif self._warpType == 'patch':
            aam = PatchAAM(frames,
                           group=self._landmarkGroup,
                           holistic_features=self._features,
                           diagonal=self._diagonal,
                           scales=self._scales,
                           max_shape_components=self._max_shape_components,
                           max_appearance_components=self._max_appearance_components,
                           patch_shape=self._extractOpts['patch_shape'],
                           verbose=False)

        else:
            raise Exception('Unknown warp type. Did you mean holistic/patch ?')

        frame_buffer = LazyList.init_from_iterable([])
        buffer_len = 256
        for idx, file in enumerate(files[1:]):
            with open('log_' + self._outModelName + '.txt', 'w') as log:
                log.write(str(idx) + ' ' + file + '\n')

            frames = mio.import_video(file,
                                      landmark_resolver=self._myresolver,
                                      normalize=True,
                                      exact_frame_count=True)
            idx_above_thresh, idx_lip_opening = landmark_filter(
                file,
                landmark_dir=self._landmarkDir,
                threshold=self._confidence_thresh,
                keep=self._kept_frames)

            frames = frames[idx_above_thresh]
            frames = frames[idx_lip_opening]
            frames = frames.map(attach_semantic_landmarks)
            if self._greyscale is True:
                frames = frames.map(convert_to_grayscale)

            frame_buffer += frames
            if len(frame_buffer) > buffer_len:
                # 2. retrain AAM
                aam.increment(frame_buffer,
                              group=self._landmarkGroup,
                              shape_forgetting_factor=1.0,
                              appearance_forgetting_factor=1.0,
                              verbose=False,
                              batch_size=None)
                del frame_buffer
                frame_buffer = LazyList.init_from_iterable([])
            else:
                pass

        if len(frame_buffer) != 0:  #
            # deplete remaining frames
            aam.increment(frame_buffer,
                          group=self._landmarkGroup,
                          shape_forgetting_factor=1.0,
                          appearance_forgetting_factor=1.0,
                          verbose=False,
                          batch_size=None)
            del frame_buffer

        mio.export_pickle(obj=aam, fp=self._outDir + self._outModelName, overwrite=True, protocol=4)

    def get_feature(self, file, process_opts=None):

        self._maybe_start_logging(file)
        self._load_landmark_fitter()

        frames = mio.import_video(file,
                                  landmark_resolver=self._myresolver,
                                  normalize=True,
                                  exact_frame_count=True)

        feat_shape = []
        feat_app = []
        feat_shape_app = []

        for frameIdx, frame in enumerate(frames):

            bounding_boxes = self._face_detect(frame)
            if len(bounding_boxes) > 0:
                initial_bbox = bounding_boxes[0]
                if self._log_errors is True:
                    gt_shape = frame.landmarks['pts_face']
                else:
                    gt_shape = None

                if isinstance(self._landmark_fitter, LucasKanadeAAMFitter):
                    result = self._landmark_fitter.fit_from_bb(
                        frame, initial_bbox,
                        max_iters=self._max_iters,
                        gt_shape=gt_shape)

                elif isinstance(self._landmark_fitter, DlibWrapper):  # DLIB fitter, doesn't have max_iters
                    result = self._landmark_fitter.fit_from_bb(
                        frame,
                        initial_bbox,
                        gt_shape=gt_shape)
                else:
                    raise Exception('incompatible landmark fitter')

                self._maybe_append_to_log(file, frameIdx, result)

                if self._shape == 'face':

                    if self._parameters == 'lk_fitting':
                        # skip the first 4 similarity params, probably not useful for classification
                        shape_param_frame = result.shape_parameters[-1][4:]
                        app_param_frame = result.appearance_parameters[-1]
                    elif self._parameters == 'aam_projection':
                        result_aam = self._projection_fitter.fit_from_shape(
                            frame,
                            result.final_shape,
                            max_iters=[0, 0, 0])

                        # TODO: analyse the case when aam true components are less than max components
                        shape_param_frame = result_aam.shape_parameters[-1][4:]
                        app_param_frame = result_aam.appearance_parameters[-1]

                    feat_shape.append(shape_param_frame)
                    feat_app.append(app_param_frame)
                    feat_shape_app.append(np.concatenate((shape_param_frame, app_param_frame)))

                elif self._shape == 'lips':

                    # extract lips landmarks from the final face fitting to initialize the part model fitting

                    aam_lips = mio.import_pickle(self._part_aam)
                    fitter_lips = LucasKanadeAAMFitter(aam_lips, lk_algorithm_cls=WibergInverseCompositional,
                                                       n_shape=[10, 20], n_appearance=[20, 150])

                    result_lips = fitter_lips.fit_from_shape(
                        image=frame,
                        initial_shape=_pointcloud_subset(result.final_shape, 'lips'),
                        max_iters=[5, 5])

                    shape_param_frame_lips = result_lips.shape_parameters[-1][4:]
                    app_param_frame_lips = result_lips.appearance_parameters[-1]

                    feat_shape.append(shape_param_frame_lips)
                    feat_app.append(app_param_frame_lips)
                    feat_shape_app.append(np.concatenate((shape_param_frame_lips, app_param_frame_lips)))

                elif self._shape == 'chin':

                    # extract chin and lips landmarks from the final face fitting to initialize the part model fitting

                    aam_chin = mio.import_pickle(self._part_aam)
                    fitter_chin = LucasKanadeAAMFitter(aam_chin, lk_algorithm_cls=WibergInverseCompositional,
                                                       n_shape=[10, 20, 25], n_appearance=[20, 50, 150])

                    result_chin = fitter_chin.fit_from_shape(
                        image=frame,
                        initial_shape=_pointcloud_subset(result.final_shape, 'chin'),
                        max_iters=[10, 10, 5])

                    shape_param_frame_mchin = result_chin.shape_parameters[-1][4:]
                    app_param_frame_mchin = result_chin.appearance_parameters[-1]

                    feat_shape.append(shape_param_frame_mchin)
                    feat_app.append(app_param_frame_mchin)
                    feat_shape_app.append(np.concatenate((shape_param_frame_mchin, app_param_frame_mchin)))

                else:
                    raise Exception('Unknown shape model, currently supported are: face, lips, chin')

            else:  # we did not detect any face

                zero_feat_shape = np.zeros(process_opts['shape_components'][-1])
                zero_feat_app = np.zeros(process_opts['appearance_components'][-1])
                zero_feat_shape_app = np.zeros(
                    process_opts['shape_components'][-1] + process_opts['appearance_components'][-1])

                feat_shape.append(zero_feat_shape)
                feat_app.append(zero_feat_app)
                feat_shape_app.append(zero_feat_shape_app)

        npfeat_shape = np.array(feat_shape)
        npfeat_app = np.array(feat_app)
        npfeat_shape_app = np.array(feat_shape_app)

        npfeat_app_delta = vsrmath.accurate_derivative(npfeat_app, 'delta')
        npfeat_shape_app_delta = vsrmath.accurate_derivative(npfeat_shape_app, 'delta')

        return {'shape': npfeat_shape,
                'app': npfeat_app,
                'shape_app': npfeat_shape_app,
                'app_delta': npfeat_app_delta,
                'shape_app_delta': npfeat_shape_app_delta}

    def _myresolver(self, file, frame):
        frames_dir = file_to_feature(str(file), extension='')
        return {'pts_face': self._landmarkDir + frames_dir + '/frame_' + str(frame + 1) + '.pts'}

    def _maybe_start_logging(self, file):
        if self._log_errors is True:
            from os import makedirs
            makedirs('./run/logs/' + self._title, exist_ok=True)
            cf = file_to_feature(file, extension='')
            with open('./run/logs/' + self._title + '/log_' + cf + '.txt', 'w') as log:
                log.write('{} \n'.format(file))
                log.write('Face detector: {}\n'.format(self._face_detect_method))
                if self._fitter_type == 'aam':
                    log.write('AAM Landmark fitter: {}\n'.format(self._aam_fitter_file))
                elif self._fitter_type == 'ert':
                    log.write('Pretrained ERT Landmark fitter\n')
                if self._parameters == 'projection':
                    log.write('AAM Projector: {}\n'.format(self._projection_aam_file))

    def _maybe_append_to_log(self, file, frame_idx, result):
        if self._log_errors is True:
            cf = file_to_feature(file, extension='')
            error = result.final_error()
            with open('./run/logs/' + self._title + '/log_' + cf + '.txt', 'a') as log:
                log.write('frame {}. error: {} \n'.format(str(frame_idx), str(error)))

    def _load_landmark_fitter(self):
        if self._fitter_type == 'aam':
            self._aam_fitter = mio.import_pickle(self._aam_fitter_file)
            fitter = LucasKanadeAAMFitter(self._aam_fitter, lk_algorithm_cls=WibergInverseCompositional,
                                          n_shape=self._shape_components, n_appearance=self._appearance_components)
        elif self._fitter_type == 'ert':
            fitter = DlibWrapper(
                path.join(current_path, '../pretrained/shape_predictor_68_face_landmarks.dat'))
        else:
            raise Exception('unknown fitter, did you mean aam/ert?')

        self._landmark_fitter = fitter

# These functions need to be redefined when importing the model
# and this adds an inconvenient when distributing it
# Only benefit is halved storage requirement.

# @ndfeature
# def float32_fast_dsift(x):
#     return fast_dsift(x).astype(np.float32)
#
#
# @ndfeature
# def float32_dsift(x):
#     return dsift(x).astype(np.float32)
#
#
# @ndfeature
# def float32_hog(x):
#     return hog(x).astype(np.float32)
#
#
# @ndfeature
# def float32_no_op(x):
#     return no_op(x).astype(np.float32)


def convert_to_grayscale(image):
    if image.n_channels == 3:
        image = image.as_greyscale()
    return image


def normalize_mean_std(input_array):
    mean = np.mean(input_array)
    std = np.std(input_array)
    output = (input_array - mean) / std
    return output


def attach_semantic_landmarks(image):
    landmarks_face = image.landmarks['pts_face']
    image.landmarks['pts_chin'] = PointCloud(np.vstack((landmarks_face.points[2:15], landmarks_face.points[48:68])))
    image.landmarks['pts_lips'] = PointCloud(landmarks_face.points[48:68])
    return image


def _pointcloud_subset(face_cloud, subset):

    if subset == 'lips':
        subset_pts = PointCloud(face_cloud.points[48:68])
    elif subset == 'chin':
        subset_pts = PointCloud(np.vstack((face_cloud.points[2:15], face_cloud.points[48:68])))
    else:
        raise Exception('unsupported landmark subset, only lips/chin are implemented')

    return subset_pts


def _preprocess(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image
