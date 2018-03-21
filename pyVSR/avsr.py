from os import makedirs, path
from .Learn import htk
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from .utils import file_to_feature


class AVSR(object):
    r""" AVSR Experiment Management class
    Its main functionalities are:
    1) extracting features from the input files and storing them in an intermediary format
    2) postprocessing the features and writing .htk feature files
    3) launching a HTK-based recognition experiment
    """
    def __init__(self, num_threads=4):
        self._nThreads = num_threads

    def extract_save_features(self,
                              files,
                              feature_type=None,
                              extract_opts=None, ):

        r"""

        Parameters
        ----------
        files : `tuple` or 'list` of video file paths
        feature_type : `dct` or `aam` are currently supported
        extract_opts : ``dict` holding the configuration for feature extraction
        output_dir

        Returns
        -------

        """
        if feature_type == 'dct':
            from .Features import dct
            extractor = dct.DCTFeature()
        elif feature_type == 'landmarks':
            from .Features import landmark
            extractor = landmark.LandmarkFeature(extract_opts)
        elif feature_type == 'aam':
            from .Features import aam
            extractor = aam.AAMFeature(extract_opts=extract_opts, output_dir=output_dir)
            extractor.extract_save_features(files)
            return
        elif feature_type == 'roi':
            from .Features import roi
            extractor = roi.ROIFeature(extract_opts=extract_opts)
        else:
            raise Exception('Unknown feature type: ' + feature_type)

        with Pool(self._nThreads) as p:
            p.map(extractor.extract_save_features, files.items())

    def process_features_write_htk(self,
                                   files,
                                   feature_dir=None,
                                   feature_type=None,
                                   process_opts=None,
                                   frame_rate=30,
                                   output_dir=None,
                                   ):
        r"""
        Writes features to .htk format
        Parameters
        ----------
        files
        feature_dir : `str`, path where the pre-processed features are stored
        feature_type : `str`, : 'dct','landmark', 'aam'
        process_opts : `dict` holding the configuration for feature processing
        frame_rate : `float`
        output_dir : `str`, path to store the extracted features

        Returns
        -------

        """
        if feature_type == 'dct':
            from .Features import dct
            processor = dct.DCTFeature(feature_dir=feature_dir)
        elif feature_type == 'aam':
            from .Features import aam
            processor = aam.AAMFeature(process_opts=process_opts)
        else:
            raise Exception('Unknown feature type: ' + feature_type)

        with Pool(self._nThreads) as p:
            p.starmap(_process_one_file,
                      zip(files.items(), repeat(processor), repeat(process_opts), repeat(output_dir), repeat(frame_rate)))


def run(train_files=None,
        test_files=None,
        hmm_states=None,
        mixtures=None,
        language_model=False,
        config_dir=None,
        experiment_name=None,
        report_results=('train', 'test'),
        num_threads=1,
        ):
    r"""

    :param train_files:
    :param test_files:
    :param hmm_states:
    :param mixtures:
    :param language_model:
    :param config_dir:
    :param experiment_name:
    :param report_results:
    :return:
    """
    makedirs('./run/', exist_ok=True)
    
    htksys = htk.HTKSys(train_files,
                        test_files,
                        hmm_states=hmm_states,
                        mixtures=mixtures,
                        language_model=language_model,
                        config_dir=config_dir,
                        report_results=report_results,
                        num_threads=num_threads)
    htksys.run()

    # TODO - wrap the code below in a function

    fld = path.join('./results/',
                    str(experiment_name),
                    str(hmm_states) + 'states/')
    makedirs(fld, exist_ok=True)

    from glob import glob
    results = glob('./run/results_*')
    from shutil import copy
    for file in results:
        copy(file, fld)


def _write_feature_to_htk(features,
                          file,
                          frame_rate):

    r""" Writes to output file the arr array with the given frame rate
    in the HTK binary format
    """

    from struct import pack
    num_frames, feature_size = np.shape(features)

    outfile = open(file, 'wb')

    num_samples = np.size(features)
    sample_period = int(1 / frame_rate * (10 ** 7))
    sample_size = feature_size * 4
    parameter_kid = 9  # user-defined htk type

    header = pack(
        '>IIHH',  # big endian, 2 uint, 2 ushort
        num_samples,
        sample_period,
        sample_size,
        parameter_kid)

    content = b''
    for val in np.nditer(features, op_dtypes=np.float32, casting='same_kind', op_flags=['readonly', 'copy']):
        content += pack('>f', val)

    outfile.write(header)
    outfile.write(content)
    outfile.close()


def _process_one_file(file, processor, process_opts, out_dir, frame_rate):
    input_file = file[0]
    output_file = file[1]

    print(input_file)

    feature_dict = processor.get_feature(input_file, process_opts)

    for feature_name, feature_value in feature_dict.items():
        makedirs(path.dirname(output_file), exist_ok=True)
        _write_feature_to_htk(features=feature_value,
                              file=output_file,
                              frame_rate=frame_rate)
