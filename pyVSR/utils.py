from os import path
import numpy as np


def file_to_feature(file, extension='.h5', tree_leaves=5):
    r"""

    Parameters
    ----------
    file
    extension
    tree_leaves: `int`, number of sub-directories that define the feature name
        For example, if file='./a/b/c/d/e/f/g.mp4' and tree_leaves=4
        then feature_name = 'c_d_e_f_g' + extension

    Returns
    -------

    """
    base, ext = path.splitext(file)

    leaves = []
    for _ in range(tree_leaves):
        base, leaf = path.split(base)
        leaves.append(leaf)

    feature_name = '_'.join(leaves[::-1]) + extension

    return feature_name


def files_to_features(files, extension=''):
    feature_list = []
    for file in files:
        feature_name = file_to_feature(file, extension=extension)
        feature_list.append(feature_name)
    return feature_list


def parse_roi_file(roi_file):
    with open(roi_file, 'r') as f:
        contents = f.readlines()

    # skip = int(contents[0])
    num_lines = len(contents)
    rois = np.empty((num_lines, 4), dtype=np.int32)
    for i, line in enumerate(contents):
        elems = line.split()
        ints = list(map(int, elems))
        rois[i, :] = np.asarray(ints)

    return rois


def read_htk_header(file):
    from struct import unpack
    f = open(file, 'rb')
    f.seek(0, 0)
    header = f.read(12)
    f.close()
    num_samples, sample_period, sample_size, parameter_kind = unpack(">iihh", header)
    num_features = sample_size // 4
    return num_samples, sample_period, sample_size, parameter_kind, num_features


def read_htk_file(file):
    from struct import unpack
    f = open(file, 'rb')
    f.seek(0, 0)
    header = f.read(12)
    num_samples, sample_period, sample_size, parameter_kind = unpack(">iihh", header)
    data = np.asarray(np.fromfile(f, '>f'), dtype=np.float32)
    veclen = sample_size // 4
    data = data.reshape(np.size(data)//veclen, veclen)
    f.close()
    return data


def read_hdf5_file(file, feature_name=None):
    import h5py
    with h5py.File(file, 'r') as f:
        if feature_name is None:
            # If user did not specify a feature
            # return by default the first one at index 0
            data = f[list(f.keys())[0]].value
        else:
            data = f[feature_name].value
    return data

def split_multiproc(files, num_threads):
    num_files = len(files)
    average_size = int(np.floor(num_files // num_threads))

    idx_start = np.arange(num_threads) * average_size
    idx_end = np.arange(1, num_threads + 1) * average_size
    idx_end[-1] = num_files

    files_batched = []
    for i in range(num_threads):
        files_batched.append(files[idx_start[i]: idx_end[i]])

    return files_batched


def write_sequences_to_mlf(sequences, file):
    with open(file, 'w') as stream:
        stream.write('#!MLF!#\n')
        for label in sequences.keys():
            label_fname = path.splitext(path.split(label)[1])[0]
            stream.write('"*/' + label_fname + '.rec"\n')
            for symbol in sequences[label]:
                if symbol != 'EOS':
                    stream.write(symbol+'\n')
                else:
                    break
            stream.write('.\n')
