from .feature import Feature
from urllib import request
import os
import cv2
import dlib
import numpy as np
from .. import utils


class ROIFeature(Feature):
    r"""
    Mouth ROI Extraction pipeline using OpenCV and dlib

    Similar functionality, but without facial alignment, exists in DCTFeature. It will
    soon get deprecated.

    Main steps:
    1) face detection - dlib mmod cnn
    2) face alignment - dlib 5 landmark prediction, alignment and cropping
    3) face shape prediction - dlib 68 landmark prediction
    4) mouth cropping - segment the aligned face around the lip coordinates (landmarks [48:68])
    """

    def __init__(self,
                 extract_opts=None,
                 output_dir=None):
        r"""

        Parameters
        ----------
        extract_opts : `dict` holding the configuration for feature extraction
            Must specify the following options:
            ``gpu`` : `boolean`, whether to use the dlib's CNN-based face detector (`True`)
                        or the traditional dlib HOG-based face detector (`False`)
            ``color`` : `boolean`, store RGB images (`True`) or grayscale images (`False`)
            ``border`` : `int`, number of pixels to pad the tightly-cropped mouth region
            ``window_size`` : `tuple` of two `ints`, one for each image dimension
                Represents the sub-sampled ROI window size, hence the full DCT matrix has the same shape
        output_dir
        """
        # if extract_opts is not None:
        #     if 'need_coords' in extract_opts:
        #         self._need_coords = extract_opts['need_coords']

        if 'window_size' not in extract_opts:
            raise Exception('window_size is mandatory')
        self._xres = extract_opts['window_size'][0]
        self._yres = extract_opts['window_size'][1]

        if 'color' in extract_opts:
            self._channels = 3 if extract_opts['color'] is True else 1

        if 'border' in extract_opts:
            self._border = extract_opts['border']
        else:
            self._border = 0

        if 'gpu' in extract_opts:
            self._gpu = extract_opts['gpu']

        if 'num_subdirs_to_file' in extract_opts:
            self._tree_leaves = extract_opts['num_subdirs_to_file']
        else:
            self._tree_leaves = 5  # default for tcdtimit

        self._output_dir = output_dir

    def extract_save_features(self, file):
        r"""

        Parameters
        ----------
        file : `str`, path to video file

        Returns
        -------

        """
        # Not all the fitters are pickleable for multiprocessing to work
        # thus load the fitters for every process
        self._preload_dlib_detector_fitter()
        roi_sequence = self.extract_roi_sequence(file)
        outfile = utils.file_to_feature(file, extension='.h5', tree_leaves=self._tree_leaves)
        self._write_sequence_to_file(outfile, roi_sequence, 'gray', (None, None, None, None))

    def extract_roi_sequence(self, file):
        stream = cv2.VideoCapture(file)
        vidframes = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        roi_seq = np.zeros((vidframes, self._xres, self._yres, self._channels), dtype=np.float32)

        current_frame = 0

        while stream.isOpened():
            ret, frame = stream.read()
            if ret is False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # dlib and opencv use different channel representations

            bbox = self._detect(frame, 0)[0]
            left, top, right, bottom = _get_bbox_corners(bbox, self._gpu)
            face_area = dlib.rectangle(left, top, right, bottom)
            landmarks5 = self._fitter5(frame, face_area)

            aligned_face = dlib.get_face_chip(frame, landmarks5, 256)
            aligned_face = np.asarray(aligned_face)

            face_chip_area = dlib.rectangle(0, 0, aligned_face.shape[0], aligned_face.shape[1])
            landmarks68 = self._fitter68(aligned_face, face_chip_area)

            arr = _dlib_parts_to_numpy(landmarks68)[48:68]
            top_left, bottom_right = _get_array_bounds(arr, aligned_face.shape, border=self._border)

            mouth_crop = aligned_face[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            mouth_crop = cv2.resize(mouth_crop, (self._xres, self._yres), cv2.INTER_CUBIC)

            if self._channels == 3:
                roi_seq[current_frame, :] = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2BGR) / 255
            else:
                roi_seq[current_frame, :] = np.expand_dims(cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY) / 255, -1)

            # Enable these when debugging
            # cv2.imshow('', roi_seq[current_frame, :])
            # cv2.waitKey(1)

            current_frame += 1

        stream.release()
        # cv2.destroyAllWindows()

        return roi_seq

    def _preload_dlib_detector_fitter(self):
        r"""
        Returns the dlib face detector and the landmark fitters (5 and 68 landmarks)
        -------

        """
        detector_path, predictor5_path, predictor68_path = maybe_download_models()

        if self._gpu is True:
            self._detect = dlib.cnn_face_detection_model_v1(detector_path)
        else:
            self._detect = dlib.get_frontal_face_detector()

        self._fitter5 = dlib.shape_predictor(predictor5_path)
        self._fitter68 = dlib.shape_predictor(predictor68_path)

    def get_feature(self, file, feat_opts):
        pass

    def _write_sequence_to_file(self, file, seq, seq_name, seq_shape):
        import h5py
        f = h5py.File(self._output_dir+file, 'w')
        f.create_dataset(seq_name, data=seq.astype('float32'),
                         maxshape=seq_shape,
                         compression="gzip",
                         fletcher32=True)
        f.close()

        return


def maybe_download_models():
    r"""
    -------

    """
    os.makedirs('./stored_models/', exist_ok=True)
    face_detector = 'https://github.com/georgesterpu/stored_models/raw/master/mmod_human_face_detector.dat'
    lm_predictor_5 = 'https://github.com/georgesterpu/stored_models/raw/master/shape_predictor_5_face_landmarks.dat'
    lm_predictor_68 = 'https://github.com/georgesterpu/stored_models/raw/master/shape_predictor_68_face_landmarks.dat'

    detector_path = './stored_models/detector.dat'
    if not os.path.isfile(detector_path):
        print('Downloading face detector')
        request.urlretrieve(face_detector, detector_path)

    predictor5_path = './stored_models/predictor5.dat'
    if not os.path.isfile(predictor5_path):
        print('Downloading landmark predictor5')
        request.urlretrieve(lm_predictor_5, predictor5_path)

    predictor68_path = './stored_models/predictor68.dat'
    if not os.path.isfile(predictor68_path):
        print('Downloading landmark predictor68')
        request.urlretrieve(lm_predictor_68, predictor68_path)

    return detector_path, predictor5_path, predictor68_path


def _dlib_parts_to_numpy(landmarks):
    parts = landmarks.parts()
    arr = []
    for part in parts:
        arr.append((part.x, part.y))

    return np.asarray(arr)


def _get_array_bounds(arr, outer_shape, border=0):
    # TODO : make border padding relative to bbox size
    top_left = np.min(arr, axis=0)
    bottom_right = np.max(arr, axis=0)

    top_left[0] = np.maximum(top_left[0] - border, 0)
    top_left[1] = np.maximum(top_left[1] - border, 0)

    bottom_right[0] = np.minimum(bottom_right[0] + border, outer_shape[0])
    bottom_right[1] = np.minimum(bottom_right[1] + border, outer_shape[1])

    return tuple(top_left), tuple(bottom_right)


def _get_bbox_corners(bbox, gpu):
    if gpu is True:
        left, top, right, bottom = bbox.rect.left(), bbox.rect.top(), bbox.rect.right(), bbox.rect.bottom()
    else:
        left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()

    return left, top, right, bottom
