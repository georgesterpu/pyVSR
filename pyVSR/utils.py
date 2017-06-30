from os import path
import numpy as np


def file_to_feature(file, extension='.h5'):
    base, ext = path.splitext(file)
    trunk, sentence = path.split(base)
    trunk, pose = path.split(trunk)
    trunk, clips = path.split(trunk)
    trunk, spkid = path.split(trunk)
    trunk, subject = path.split(trunk)
    feature_name = subject + '_' + spkid + '_' + clips + '_' + pose + '_' + sentence + extension
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
    num_samples, sample_period, sample_size, parameter_kind = unpack(">IIHH", header)
    num_features = sample_size // 4
    return num_samples, sample_period, sample_size, parameter_kind, num_features


def read_htk_file(file):
    from struct import unpack
    f = open(file, 'rb')
    f.seek(0, 0)
    header = f.read(12)
    num_samples, sample_period, sample_size, parameter_kind = unpack(">IIHH", header)
    data = np.asarray(np.fromfile(f, '>f'), dtype=np.float32)
    veclen = sample_size // 4
    data = data.reshape(np.size(data)//veclen, veclen)
    f.close()
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
            stream.write('.\n')
