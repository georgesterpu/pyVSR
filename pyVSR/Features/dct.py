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
                 output_dir=None):
        r"""

        Parameters
        ----------
        extract_opts : `dict` holding the configuration for feature extraction
            Must specify the following options:
            ``roi_extraction`` : `stored` or `dlib`

                When using `stored` mode, the path to the directory storing the ROI for each file
                has to be specified in the subsequent option `roi_dir`
                The ROI dir should contain the feature-like file names with .roi extensions
                (e.g. .utils.file_to_feature(video_file, extension='.roi'). Every line in a file represents
                the ROI coordinates in a frame and contains for numbers X,Y, DX, DY, where X, Y are the
                top left coordinates of the ROI, and DX, DY are the width and height of the box

                When using `dlib` mode, the ROI coordinates are computed on the fly, first by detecting a set of
                landmarks with the Dlib's pre-trained ERT shape predictor, then cropping the region around
                the lips. An extra option `boundary_proportion` has to be specified, inflating the area
                around the lips by some amount.

            ``roi_dir`` : `str`, directory storing ROI coordinates, must be used in conjunction with `stored`
            ``boundary_proportion`` : positive `float`, used in conjuction with `dlib`
            ``window_size`` : `tuple` of two `ints`, one for each image dimension
                Represents the sub-sampled ROI window size, hence the full DCT matrix has the same shape

        output_dir : `str`, absolute path where the features are to be stored
        """
        if extract_opts is not None:
            self._featOpts = extract_opts
            self._roiExtraction = extract_opts['roi_extraction']
            if self._roiExtraction == 'stored':
                self._roiDir = extract_opts['roi_dir']
            elif self._roiExtraction == 'dlib':
                self._boundary_proportion = extract_opts['boundary_proportion']
            else:
                raise Exception('The supported methods for ROI extraction are: stored, dlib')
            self._xres = extract_opts['window_size'][0]
            self._yres = extract_opts['window_size'][1]
        self._featDir = output_dir

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
        dct = self._load_dct(file)
        rows, cols, frames = dct.shape

        mask = process_opts['mask']
        keep_basis = process_opts['keep_basis']
        delta = process_opts['delta']
        double_delta = process_opts['double_delta']
        derivative_order = process_opts['derivative_order']

        interp_factor = process_opts['interpolation_factor']

        ncoeffs, first, last = _parse_mask(mask)

        dct_zz = np.zeros((frames, ncoeffs))

        for frame in range(frames):
            frame_dct = dct[:, :, frame]
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

    def _load_dct(self, file):
        import h5py
        with h5py.File(self._featDir + file, 'r') as f:
            dct_seq = f['dct'].value
        return dct_seq

    def extract_save_features(self, file):
        r"""

        Parameters
        ----------
        file

        Returns
        -------

        """
        dct = self._compute_3d_dct(file)
        outfile = utils.file_to_feature(file, extension='.h5')
        self._write_dctseq_to_file(outfile, dct)

    def _compute_3d_dct(self, file):
        r"""
        Video import is based on menpo implementation
        (LazyList with ffmpeg backend)
        :param file:
        :return:
        """
        from menpo.io import import_video
        frames = import_video(filepath=file, normalize=True, exact_frame_count=True)
        dct_volume = np.zeros((self._yres, self._xres, len(frames)), dtype=np.float32)

        if self._roiExtraction == 'stored':
            self._rois = self._get_frames_rois(file)

        for frame_idx, frame in enumerate(frames):
            roi = self._get_roi(frame, frame_idx)
            dctmat = np.zeros(np.shape(roi))
            cv2.dct(roi, dctmat)
            dct_volume[:, :, frame_idx] = dctmat

        return dct_volume

    def _get_roi(self, menpo_image, frame_idx):
        r"""
        Extracts the mouth ROI from a menpo.image.Image object
        :param menpo_image:
        :param frame_idx:
        :return:
        """
        if self._roiExtraction == 'stored':
            bounds = self._rois[frame_idx]
            gray = menpo_image.as_greyscale()
            gray = gray.crop(
                min_indices=(bounds[1], bounds[0]),
                max_indices=(bounds[1] + bounds[3], bounds[0] + bounds[2])
            )
            gray = gray.resize((self._xres, self._yres))
            roi = gray.pixels[0]
        elif self._roiExtraction == 'dlib':
            roi_pointcloud, success = _get_roi_pointcloud(menpo_image)
            if success:
                roi = menpo_image.crop_to_pointcloud_proportion(
                    roi_pointcloud,
                    boundary_proportion=self._boundary_proportion)
                roi = roi.as_greyscale()
                roi = roi.resize((self._xres, self._yres))
                roi = roi.pixels[0]  # why do we have a shape of 1xMxN ?
            else:
                roi = np.zeros((self._xres, self._yres))
        else:
            raise Exception('Normally an unreachable use-case')

        return roi

    def _write_dctseq_to_file(self, file, dct_seq):
        import h5py
        f = h5py.File(self._featDir+file, 'w')
        f.create_dataset('dct', data=dct_seq.astype('float32'),
                         maxshape=(None, None, None),
                         compression="gzip",
                         fletcher32=True)
        f.close()

        return

    def _get_frames_rois(self, file):
        roif = utils.file_to_feature(file, extension='.roi')
        rois = utils.parse_roi_file(self._roiDir+roif)
        return rois

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

        rois = self._get_frames_rois(file)

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


def zz(matrix, nb):
    """Zig-zag traversal of the input matrix
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
    """ Reconstruction of a matrix from zig-zag coefficients
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
    # numpy array indexing: lines are first index => y direction goes first
    cropped = fullframe[ypos:ypos+ylen, xpos:xpos+xlen]
    return cropped


def _parse_mask(mask):
    tmp = mask.split('-')
    first = int(tmp[0])
    last = int(tmp[1])
    ncoeffs = last-first+1
    return ncoeffs, first, last


def _get_roi_pointcloud(image):
    from menpofit.dlib import DlibWrapper
    from menpodetect import load_dlib_frontal_face_detector
    from menpo.shape import PointCloud
    from os import path
    dir_ = path.dirname(__file__)
    fitter = DlibWrapper(path.join(dir_, '../pretrained/shape_predictor_68_face_landmarks.dat'))
    detect = load_dlib_frontal_face_detector()

    bboxes = detect(image)
    pcld = PointCloud([])
    if len(bboxes) >= 1:
        result = fitter.fit_from_bb(image, bounding_box=bboxes[0])
        pcld = PointCloud(result.final_shape.points[48:68])
        success = True
    else:  # no face detected
        success = False

    return pcld, success
