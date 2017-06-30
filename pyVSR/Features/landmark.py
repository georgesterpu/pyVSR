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

    def __init__(self, files, opts, out_dir):

        self._files = files
        self._featOpts = opts
        self._outDir = out_dir
        self._binFile = path.join(current_path, '../bins/openface/FeatureExtraction')

        self._nLandmarks = 68

    def extract_save_features(self, file):
        self._find_landmarks_openface(file)
        self._parse_landmarks_file(file)

    def get_feature(self, file, feat_opts):
        pass

    def _find_landmarks_openface(self, file):
        outfile = utils.file_to_feature(file, extension='.full.pts')

        cmd = [self._binFile, '-f', file, '-of', self._outDir + outfile, '-q',
               '-noAUs', '-noGaze', '-noPose', '-no3Dfp', '-noMparams']
        sp.run(cmd, check=True, stdout=sp.PIPE)

    def _parse_landmarks_file(self, file):
        r"""
        :param file:
        :return:
        """

        infile = utils.file_to_feature(file, extension='.full.pts')
        file_dir = utils.file_to_feature(file, extension='/')
        makedirs(self._outDir + file_dir, exist_ok=True)

        # get openface file
        with open(self._outDir + infile, 'r') as f:
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

            with open(self._outDir + file_dir + 'frame_' + str(idx+1) + '.pts', 'w') as g:
                g.writelines(write_buffer)


def landmark_filter(file, landmark_dir, threshold, keep):

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
