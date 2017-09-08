from .feature import Feature
from .. import utils
import cv2
import numpy as np
from .. import vsrmath


class DCTFeature(Feature):
    r"""
    Discrete Cosine Transform (DCT) feature extraction
    """

    def __init__(self,
                 extract_opts=None,
                 feature_dir=None):
        r"""

        Parameters
        ----------
        extract_opts : `dict` holding the configuration for feature extraction
            Must specify the following options:
            ``roi_extraction`` : `coords`, `dct`, `rgb` or `gray`

                Extracts ROI cordinates (`coords`), ROI pixels in RGB (`rgb`) or converted to grayscale (`gray`),
                or directly DCT coefficients (`dct`) from the grayscale ROI

            ``need_coords``: `Boolean`.

                If `True`, the ROI coordinates are computed on the fly, first by detecting a set of
                landmarks with the Dlib's pre-trained ERT shape predictor, then cropping the region around
                the lips. An extra option `boundary_proportion` has to be specified, inflating the area
                around the lips by some amount. The coordinates will be stored at the `roi_dir` path.
                For subsequent runs, `roi_extraction` can then be set to `no`.
                Important note: the pre-trained ERT model is not distributed with pyVSR and has to be downloaded
                manually from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                then uncompressed to ./pyVSR/pretrained/

                If `False`, the path to the directory storing the ROI for each file
                has to be specified in the subsequent option `roi_dir`
                The ROI dir should contain the feature-like file names with .roi extensions
                (e.g. .utils.file_to_feature(video_file, extension='.roi'). Every line in a file represents
                the ROI coordinates in a frame and contains for numbers X,Y, DX, DY, where X, Y are the
                top left coordinates of the ROI, and DX, DY are the width and height of the box.

            ``roi_dir`` : `str`, directory storing ROI coordinates

            ``boundary_proportion`` : positive `float`, used in conjuction with `dlib`

            ``window_size`` : `tuple` of two `ints`, one for each image dimension
                Represents the sub-sampled ROI window size, hence the full DCT matrix has the same shape

            ``video_backend`` : `menpo` or `opencv`
                The `menpo` backend is based on ffmpeg and should work out of the box on most platforms.
                The `opencv` backend is much faster, but you may have problems to properly set it up under Anaconda
                (some mystery around video support in cv2.VideoCapture, you don't know it from here)

        feature_dir : `str`, directory where the features in .h5 format are stored
        """
        if extract_opts is not None:
            self._featOpts = extract_opts
            self._roiExtraction = extract_opts['roi_extraction']
            self._roiDir = extract_opts['roi_dir']

            if 'need_coords' in extract_opts:
                self._need_coords = extract_opts['need_coords']
            else:
                self._need_coords = False

            if self._need_coords is True:
                self._boundary_proportion = extract_opts['boundary_proportion']

            if 'video_backend' in extract_opts:
                self._video_backend = extract_opts['video_backend']
            else:
                # `menpo` backend enabled by default as `opencv` needs to be compiled with ffmpeg flag
                self._video_backend = 'menpo'

            self._xres = extract_opts['window_size'][0]
            self._yres = extract_opts['window_size'][1]
        self._featDir = feature_dir

    def get_feature(self, file, process_opts):
        r"""

        Parameters
        ----------
        file: `str`, path to a .h5 file storing the DCT matrix
        process_opts : `dict` holding the post-processing configuration
            Must specify the following options:
            ``mask`` : `str`, specifies which DCT coefficients should be kept
                Example: a mask of '1-44` will keep 44 coefficients, discarding the bias term
                The coefficients are chosen in a zig-zag pattern from the top-left corner
            ``keep_basis`` : `boolean`, states whether final feature should contain the DCT coeffs
            ``delta`` : `boolean`, states whether final feature should contain the first derivative of the DCT coeffs
            ``double_delta`` : `boolean`, state whether the final feature should contain the second derivative
            ``derivative_order`` : `first` or `fourth`, accuracy of the derivatives
                Central differences schemes are used (but forward and backward at the boundaries)
                The derivative window length is minimal for the requested order of accuracy
                (e.g. two forward and two backward coefficients for the fourth order accurate central derivative)
            ``interpolation_factor`` : positive `int`, the new sampling rate is a multiple of this factor
            ``interpolation_type`` : `str`, must be used in conjunction with non-unitary `interpolation_factor`s
                It is the same as the `kind` parameter of scipy.interpolate.interp1d
                Some options: `linear`, `quadratic`, `cubic`
        Returns
        -------
        A dictionary of one key: `DCT`, holding the DCT-based feature
        """
        dct = self._load_h5_file(file, 'dct')
        frames, rows, cols = dct.shape

        mask = process_opts['mask']
        keep_basis = process_opts['keep_basis']
        delta = process_opts['delta']
        double_delta = process_opts['double_delta']
        derivative_order = process_opts['derivative_order']

        interp_factor = process_opts['interpolation_factor']

        ncoeffs, first, last = _parse_mask(mask)

        dct_zz = np.zeros((frames, ncoeffs))

        for frame in range(frames):
            frame_dct = dct[frame, :, :]
            frame_dct_truncated = zz(frame_dct, ncoeffs+first)
            # beware of python indexing, user mask is all-inclusive
            dct_zz[frame, :] = frame_dct_truncated[first:(last+1)]

        if interp_factor != 1:
            interp_kind = process_opts['interpolation_kind']
            dct_zz_interp = vsrmath.interpolate_feature(
                feat=dct_zz,
                factor=interp_factor,
                interpolation_kind=interp_kind)
            dct_zz = dct_zz_interp  # make it more obvious that we're changing the reference

        feature = dct_zz

        if delta is True:
            if derivative_order == 'first':
                # delta_coeffs_dummy = self._accurate_derivative(dct_zz, type='delta')
                delta_coeffs = np.diff(dct_zz, n=1, axis=0)
                delta_coeffs = np.vstack((delta_coeffs, delta_coeffs[-1, :]))
            elif derivative_order == 'fourth':
                delta_coeffs = vsrmath.accurate_derivative(dct_zz, derivative_type='delta')
            else:
                raise Exception('Unsupported derivative order')
            feature = np.hstack((feature, delta_coeffs))

        if double_delta is True:
            if derivative_order == 'first':
                accel_coeffs = np.diff(dct_zz, n=2, axis=0)
                accel_coeffs = np.vstack((accel_coeffs, accel_coeffs[-1, :], accel_coeffs[-1, :]))
            elif derivative_order == 'fourth':
                accel_coeffs = vsrmath.accurate_derivative(dct_zz, derivative_type='double_delta')
            else:
                raise Exception('Unsupported derivative order')
            feature = np.hstack((feature, accel_coeffs))

        if keep_basis is False:
            feature = feature[:, ncoeffs:]  # discards first ncoeffs cols (the dct features)

        return {'DCT': feature}

    def _load_h5_file(self, file, feature_name):
        import h5py
        with h5py.File(self._featDir + file, 'r') as f:
            dct_seq = f[feature_name].value
        return dct_seq

    def extract_save_features(self, file):
        r"""

        Parameters
        ----------
        file

        Returns
        -------

        """
        if self._need_coords is True:
            self._preload_dlib_detector_fitter()
            self._detect_write_roi_bounds(file)

        if self._roiExtraction == 'coords':
            return  # job already done

        elif self._roiExtraction == 'dct':

            dct = self._compute_3d_dct(file)
            outfile = utils.file_to_feature(file, extension='.h5')
            self._write_sequence_to_file(outfile, dct, 'dct', (None, None, None))

        elif self._roiExtraction == 'rgb':
            rgb = self._get_rois_opencv(file, mode='rgb')
            outfile = utils.file_to_feature(file, extension='.h5')
            self._write_sequence_to_file(outfile, rgb, 'rgb', (None, None, None, 3))

        elif self._roiExtraction == 'gray':
            gray = self._get_rois_opencv(file, mode='gray')
            outfile = utils.file_to_feature(file, extension='.h5')
            self._write_sequence_to_file(outfile, gray, 'gray', (None, None, None))

        else:
            return

    def _detect_write_roi_bounds(self, file):
        from menpo.io import import_video
        frames = import_video(filepath=file, normalize=True, exact_frame_count=True)

        bounds = []
        for frame_idx, frame in enumerate(frames):

            lips_pointcloud, success = _get_roi_pointcloud(frame, self._detect, self._fitter)
            if success is True:
                roi_bounds = _get_pointcloud_bounds(lips_pointcloud, self._boundary_proportion)
            else:
                roi_bounds = [-1, -1, -1, -1]
            bounds.append(roi_bounds)
        self._write_rois_to_file(file, bounds)

    def _compute_3d_dct(self, file):
        if self._video_backend == 'menpo':
            self._preload_dlib_detector_fitter()
            dct = self._compute_3d_dct_menpo(file)
        elif self._video_backend == 'opencv':
            dct = self._compute_3d_dct_opencv(file)
        else:
            raise Exception('Available video backends are `menpo` and `opencv`')

        return dct

    def _compute_3d_dct_menpo(self, file):
        r"""
        Video import is based on menpo implementation
        (LazyList with ffmpeg backend)
        :param file:
        :return:
        """
        from menpo.io import import_video
        frames = import_video(filepath=file, normalize=True, exact_frame_count=True)
        dct_volume = np.zeros((self._yres, self._xres, len(frames)), dtype=np.float32)

        for frame_idx, frame in enumerate(frames):
            roi = self._get_roi(frame, frame_idx, file)

            dctmat = np.zeros(np.shape(roi))
            cv2.dct(roi, dctmat)
            dct_volume[:, :, frame_idx] = dctmat

        return dct_volume

    def _get_roi(self, menpo_image, frame_idx, roi_file):
        r"""
        Extracts from a video frame the lips ROI delimited by the bounds
        specified in the ROI file
        Parameters
        ----------
        menpo_image
        frame_idx
        roi_file

        Returns
        -------

        """
        video_roi = self._read_roi_file(roi_file)
        x_min, y_min, x_delta, y_delta = video_roi[frame_idx]
        try:
            cropped = menpo_image.crop(
                min_indices=(y_min, x_min),
                max_indices=(y_min + y_delta, x_min + x_delta)
            )
        except Exception as e:
            print('Exception: ' + str(e) + '. Filling ROI with zeros.')
            return np.zeros((self._xres, self._yres))

        gray = cropped.as_greyscale()
        gray = gray.resize((self._xres, self._yres))
        roi = gray.pixels[0]

        return roi

    def _write_sequence_to_file(self, file, seq, seq_name, seq_shape):
        import h5py
        f = h5py.File(self._featDir+file, 'w')
        f.create_dataset(seq_name, data=seq.astype('float32'),
                         maxshape=seq_shape,
                         compression="gzip",
                         fletcher32=True)
        f.close()

        return

    def _read_roi_file(self, file):
        r"""
        Reads the contents of a ROI file
        Parameters
        ----------
        file

        Returns
        -------

        """
        roif = utils.file_to_feature(file, extension='.roi')
        rois = utils.parse_roi_file(self._roiDir+roif)
        return rois

    def _write_rois_to_file(self, file, bounds):
        roif = utils.file_to_feature(file, extension='.roi')
        from os import makedirs, path

        makedirs(self._roiDir, exist_ok=True)

        buffer = ''
        for line in bounds:
            buffer += ' '.join(str(x) for x in line) + '\n'

        with open(path.join(self._roiDir, roif), 'w') as f:
            f.write(buffer)

    def _get_rois_opencv(self, file, mode='gray'):
        cap = cv2.VideoCapture(file)
        vidframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rois = self._read_roi_file(file)
        totalframes = rois.shape[0]

        if totalframes != vidframes:
            print('Roi Frames: %d\n' % totalframes)
            print('Vid Frames: %d\n' % vidframes)
            raise Exception('Mismatch between the actual number of video frames and the provided ROI _labels')

        if mode == 'gray':
            roi_seq = np.zeros((totalframes, self._yres, self._xres), dtype=np.float32)
        elif mode == 'rgb':
            roi_seq = np.zeros((totalframes, self._yres, self._xres, 3), dtype=np.float32)
        else:
            raise Exception('gray or rgb')

        this_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            if mode == 'gray':
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_roi = _crop_roi(gray, rois[this_frame, :])
                resized = self._resize_frame(gray_roi)
            elif mode == 'rgb':
                rgb_roi = _crop_roi(frame, rois[this_frame, :])
                resized = self._resize_frame(rgb_roi)
            else:
                raise Exception('gray or rgb')

            roi_seq[this_frame, :, :] = resized

            this_frame += 1

        return roi_seq

    def _compute_3d_dct_opencv(self, file):

        r"""
        Runs much faster than the menpo-based one
        but usually opencv is distributed without video support (compile flag)
        and is harder to set up
        Works fine with the opencv package from arch linux repos
        in which case the system Python has to be used
        :param file:
        :return:
        """
        cap = cv2.VideoCapture(file)
        vidframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        rois = self._read_roi_file(file)

        totalframes = rois.shape[0]

        if totalframes != vidframes:
            print('Roi Frames: %d\n' % totalframes)
            print('Vid Frames: %d\n' % vidframes)
            raise Exception('Mismatch between the actual number of video frames and the provided ROI _labels')

        dct_seq = np.zeros((self._yres, self._xres, totalframes),
                           dtype=np.float32)  # _yres goes first since numpy indexing is rows-first

        this_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_roi = _crop_roi(gray, rois[this_frame, :])
            resized = self._resize_frame(gray_roi)

            dctmat = np.zeros(np.shape(resized))
            cv2.dct(resized, dctmat)

            dct_seq[:, :, this_frame] = dctmat

            this_frame += 1

        return dct_seq

    def _resize_frame(self, frame):
        resized = cv2.resize(frame, (self._xres, self._yres), interpolation=cv2.INTER_CUBIC) / 255
        return resized

    def _preload_dlib_detector_fitter(self):
        from menpofit.dlib import DlibWrapper
        from menpodetect import load_dlib_frontal_face_detector

        from os import path
        dir_ = path.dirname(__file__)
        self._fitter = DlibWrapper(path.join(dir_, '../pretrained/shape_predictor_68_face_landmarks.dat'))
        self._detect = load_dlib_frontal_face_detector()


def zz(matrix, nb):
    r"""Zig-zag traversal of the input matrix
    :param matrix: input matrix
    :param nb: number of coefficients to keep
    :return: an array of nb coefficients
    """
    flipped = np.fliplr(matrix)
    rows, cols = flipped.shape  # nb of columns

    coefficient_list = []

    for loop, i in enumerate(range(cols - 1, -rows, -1)):
        anti_diagonal = np.diagonal(flipped, i)

        # reversing even diagonals prioritizes the X resolution
        # reversing odd diagonals prioritizes the Y resolution
        # for square matrices, the information content is the same only when nb covers half of the matrix
        #  e.g. [ nb = n*(n+1)/2 ]
        if loop % 2 == 0:
            anti_diagonal = anti_diagonal[::-1]  # reverse anti_diagonal

        coefficient_list.extend([x for x in anti_diagonal])

    # flattened = [val for sublist in coefficient_list for val in sublist]
    return coefficient_list[:nb]


def izz(feature, shape):
    r""" Reconstruction of a matrix from zig-zag coefficients
    :param feature: an array of coefficients as input
    :param shape: a tuple with the number of rows and columns of the output matrix
    :return: a matrix reconstructed from zig-zag coefficients with the specified shape
    """

    reconstruction_dct = np.zeros(shape)
    rows, cols = shape[0], shape[1]

    dim1 = min(rows, cols)
    dim2 = max(rows, cols)

    feature_length = np.size(feature)
    i, j = np.indices(shape)

    prev = 0
    end = False
    tail = np.zeros(0)
    for diagonal_index in range(dim1):  # diagonals of increasing size
        first = prev
        last = prev + diagonal_index + 1
        prev += diagonal_index + 1
        if prev > feature_length:
            last = feature_length
            tail = np.zeros(prev - feature_length)
            end = True

        diag = np.append(feature[first:last], tail)

        if diagonal_index % 2 == 0:
            diag = diag[::-1]

        reconstruction_dct[i == (j - cols + 1 + diagonal_index)] = diag

        if end is True:
            break

    if end is False:
        for diagonal_index in range(dim1, dim2):  # constant size diagonals
            first = prev
            last = prev + dim1
            prev += dim1
            if prev > feature_length:
                last = feature_length
                tail = np.zeros(prev - feature_length)
                end = True

            diag = np.append(feature[first:last], tail)

            if diagonal_index % 2 == 0:
                diag = diag[::-1]

            reconstruction_dct[i == (j - cols + 1 + diagonal_index)] = diag

            if end is True:
                break

    if end is False:
        for diagonal_index in range(dim2, cols + rows - 1):
            first = prev
            last = prev + rows + cols - diagonal_index - 1
            prev += rows + cols - diagonal_index - 1
            if prev > feature_length:
                last = feature_length
                tail = np.zeros(prev - feature_length)
                end = True

            diag = np.append(feature[first:last], tail)

            if diagonal_index % 2 == 0:
                diag = diag[::-1]

            reconstruction_dct[i == (j - cols + 1 + diagonal_index)] = diag

            if end is True:
                break

    return np.fliplr(reconstruction_dct)


def _crop_roi(fullframe, roisz):
    xpos = roisz[0]
    ypos = roisz[1]
    xlen = roisz[2]
    ylen = roisz[3]
    # numpy array indexing: lines are the first index => y direction goes first

    chan = np.ndim(fullframe)

    if xpos == -1:
        cropped = np.zeros((36, 36))
    else:
        if chan == 2:
            cropped = fullframe[ypos:ypos+ylen, xpos:xpos+xlen]
        elif chan == 3:
            cropped = fullframe[ypos:ypos + ylen, xpos:xpos + xlen, :]
        else:
            raise Exception('unsupported nb of channels')

    return cropped


def _parse_mask(mask):
    r"""
    Interprets a string mask to return the number of coefficients to be kept
    and the indices of the first and last ones in the zig-zagged flattened DCT matrix
    Example: '1-44' returns 44, first=1, last=44
    Parameters
    ----------
    mask

    Returns
    -------

    """
    tmp = mask.split('-')
    first = int(tmp[0])
    last = int(tmp[1])
    ncoeffs = last-first+1
    return ncoeffs, first, last


def _get_roi_pointcloud(image, detect, fitter):
    r"""
    Predicts a set of 68 facial landmarks from a given image
    and selects the subset of the lip region
    Parameters
    ----------
    image: menpo image
    detect: face detector
    fitter: landmark fitter

    Returns
    -------
    The point cloud of lip landmarks
    A `success` boolean flag, is ``False`` if no face was detected
    """
    from menpo.shape import PointCloud

    bboxes = detect(image)

    pcld = PointCloud([])
    if len(bboxes) >= 1:
        result = fitter.fit_from_bb(image, bounding_box=bboxes[0])

        pcld = PointCloud(result.final_shape.points[48:68])
        success = True
    else:  # no face detected
        success = False

    return pcld, success


def _get_pointcloud_bounds(pointcloud, boundary_proportion):
    r"""

    Parameters
    ----------
    pcld

    Returns
    -------

    """
    bounds = pointcloud.bounds(np.min(pointcloud.range() * boundary_proportion))
    y_min, x_min = bounds[0]
    y_delta, x_delta = bounds[1] - bounds[0]
    return [int(x_min), int(y_min), int(x_delta), int(y_delta)]
