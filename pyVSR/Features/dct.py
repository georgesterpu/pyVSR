from .feature import Feature
import cv2
import numpy as np
from .. import vsrmath


def _load_h5_file(file, feature_name):
    import h5py
    with h5py.File(file, 'r') as f:
        dct_seq = f[feature_name].value
    return dct_seq


class DCTFeature(Feature):
    r"""
    Discrete Cosine Transform (DCT) feature extraction
    """

    def __init__(self):
        pass

    def get_feature(self, file, process_opts):
        r"""

        Parameters
        ----------
        file: `str`, path to a .h5 file storing the DCT/Data matrix
        process_opts : `dict` holding the post-processing configuration
            Must specify the following options:
            ``compute_dct`` : `boolean`, applies the DCT transform on the stored data
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

        data = _load_h5_file(file, 'sequence')
        if process_opts['compute_dct'] is True:
            dct = _compute_frame_dct(data)
        else:
            dct = data
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

    def extract_save_features(self, files):
        pass


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


def _compute_frame_dct(data):

    num_frames = data.shape[0]
    num_channels = data.shape[-1]

    dct = np.zeros(shape=data.shape[:-1])

    for i in range(num_frames):
        frame = data[i, :]
        if num_channels == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.squeeze(frame)

        dct[i, :, :] = cv2.dct(gray)

    return dct
