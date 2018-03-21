from .feature import Feature
from ..import utils
import subprocess as sp
from os import makedirs, path
import numpy as np

current_path = path.abspath(path.dirname(__file__))


class LandmarkFeature(Feature):
    r"""
    Facial landmarks feature extraction
    Currently used to label images for AAMs
    """

    def __init__(self, opts):
        r"""

        Parameters
        ----------
        opts
        """
        self._featOpts = opts
        self._binFile = path.join(current_path, '../bins/openface/FeatureExtraction')

        self._nLandmarks = 68

    def extract_save_features(self, example):
        r"""
        Saves the facial landmarks in text format
        Important note: the OpenFace binaries must be placed in ./bins/openface/
        Instructions here: https://github.com/TadasBaltrusaitis/OpenFace

        Parameters
        ----------
        example

        Returns
        -------

        """
        self._find_landmarks_openface(example)
        self._parse_landmarks_file(example[1])

    def get_feature(self, file, feat_opts):
        pass

    def _find_landmarks_openface(self, example):
        input_file = example[0]
        output_file = example[1]

        cmd = [self._binFile, '-f', input_file, '-of', output_file, '-q',
               '-2Dfp']

        sp.run(cmd, check=True, stdout=sp.PIPE)

    def _parse_landmarks_file(self, file):
        r"""
        For each video frame it creates a landmark file that could be interpreted by menpo
        Parameters
        ----------
        file

        Returns
        -------

        """
        with open(file, 'r') as f:
            contents = f.read().splitlines()

        header = contents[0].split(', ')
        start = 0

        for idx in range(len(header)):
            if header[idx] == 'x_0':
                start = idx
                break

        end = start + 2*68

        pts_header = ['version: 1\n', 'n_points:  ' + str(self._nLandmarks) + '\n', '{\n']

        # write a PTS file for each frame
        for idx, line in enumerate(contents[1:]):  # skip header
            write_buffer = []
            write_buffer.extend(pts_header)
            elements = line.split(', ')[start:end]

            for i in range(self._nLandmarks):
                write_buffer.append(elements[i] + ' ' + elements[i + self._nLandmarks] + '\n')

            write_buffer.append('}\n')

            outfile = path.join(
                path.dirname(file),
                path.basename(path.splitext(file)[0]), 'frame_' + str(idx+1) + '.pts')

            makedirs(path.dirname(outfile), exist_ok=True)

            with open(outfile, 'w') as g:
                g.writelines(write_buffer)


def landmark_filter(file, landmark_dir, threshold, keep):
    r"""
    From a video file, it keeps only the frames above a landmark confidence threshold
    A further uniform sampling is done to retain only the `keep` proportion of images
    Parameters
    ----------
    file
    landmark_dir
    threshold
    keep

    Returns
    -------

    """

    pts_file = utils.file_to_feature(file, extension='.full.pts')

    with open(landmark_dir + pts_file, 'r') as f:
        contents = f.read().splitlines()

    frames_idx_above_thresh = _landmark_confidence_filter(contents, threshold)

    if len(frames_idx_above_thresh) > 50:
        frames_idx_lip_opening = _landmark_sample_filter(contents, frames_idx_above_thresh, keep)
    elif len(frames_idx_above_thresh) > 21:
        frames_idx_lip_opening = _landmark_sample_filter(contents, frames_idx_above_thresh, keep)
    else:
        frames_idx_lip_opening = list(range(len(frames_idx_above_thresh)))[::10]

    return frames_idx_above_thresh, frames_idx_lip_opening


def _landmark_confidence_filter(contents, threshold):
    r"""
    Filters the frames according to a landmark confidence threshold
    Parameters
    ----------
    contents
    threshold

    Returns
    -------

    """

    confidence_id = 2
    success_id = 3

    filtered = []

    for idx, line in enumerate(contents[1:]):  # skip header
        cnf, success = line.split(', ')[confidence_id:success_id+1]
        cnf = float(cnf)
        success = int(success)
        if cnf >= threshold and success == 1:
            filtered.append(idx)

    return filtered


def _landmark_sample_filter(contents, frames_idx, keep):
    r"""
    Sampling at equally spaced intervals
    on the list of sorted frames by the amount of lip opening
    Parameters
    ----------
    contents
    frames_idx
    keep

    Returns
    -------

    """

    contents = [contents[i+1] for i in frames_idx]

    left = 48 + 4
    right = 54 + 4
    top = 51 + 68 + 4
    bot = 57 + 68 + 4

    pts = [left, right, top, bot]

    arr = np.zeros((len(contents), 2))

    for idx, line in enumerate(contents):  # no header this time
        line = line.split(', ')
        x48, x54, y51, y57 = [line[i] for i in pts]

        x48 = float(x48)
        x54 = float(x54)
        y51 = float(y51)
        y57 = float(y57)

        width = abs(x48 - x54)
        height = abs(y51 - y57)

        arr[idx][0] = width
        arr[idx][1] = height

    sorted_idx = np.argsort(arr[:, 1], axis=0)
    step_vertical = int(1 / keep)
    # step_horizontal = int(1 / (keep*0.3))  # perhaps useful to consider the width too
    sampled = np.sort(sorted_idx[::step_vertical])

    return sampled
