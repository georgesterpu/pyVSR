import numpy as np


def accurate_derivative(mat, derivative_type):
    r"""
    Computes an accurate derivative of the 1D input signal
    Supported derivatives: first, second
    The derivatives are fourth order accurate, using the minimal filter length
    """
    rows, _ = mat.shape
    filter_matrix = make_fourth_order(rows, derivative_type)
    filtered = filter_matrix @ mat
    return filtered


def make_fourth_order(rows, derivative_type='delta'):
    r"""
    Dynamically generates a filtering matrix from kernels used in
    computing the first or second derivatives that are fourth order accurate
    """
    if derivative_type == 'delta':
        filter_kernel = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        filter_forward = np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])
        filter_backward = -filter_forward  # flip sign for odd order derivative
    elif derivative_type == 'double_delta':
        filter_kernel = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
        filter_forward = np.array([15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])
        filter_backward = filter_forward  # DON'T flip sign for even order derivative
    else:
        raise Exception('Unknown derivative derivative_type')

    if np.size(filter_kernel) > rows:
        print('ERROR: Number of inputs (%d) too low for the length of the filter_kernel (%d)'
              % (rows, np.size(filter_kernel)))
        raise Exception('ERROR: Failed to compute the first derivative')

    central = np.zeros((rows - np.size(filter_kernel) + 1, rows))
    i, j = np.indices(np.shape(central))

    for idx, coefficient in enumerate(filter_kernel):
        central[i == j - idx] = coefficient

    forward = np.zeros(rows)
    forward[0:np.size(filter_forward)] = filter_forward

    backward = np.zeros(rows)
    backward[-1:-np.size(filter_backward) - 1:-1] = filter_backward  # flip the

    filter_matrix = np.vstack((forward, np.roll(forward, 1), central, np.roll(backward, -1), backward))
    return filter_matrix


def interpolate_feature(feat, factor, interpolation_kind):
    r"""
    Upsamples a signal by an integer factor
    Makes use of scipy.interp
    """
    from scipy import interpolate as interp
    frames, _ = feat.shape
    x = factor * np.arange(frames)
    interpolated_x = np.arange(x[-1])  # fill in a new index vector from 0 to Last elem in x
    f = interp.interp1d(x, feat, kind=interpolation_kind, axis=0)
    dct_zz_interp = f(interpolated_x)
    return dct_zz_interp
