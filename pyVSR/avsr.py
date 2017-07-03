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
    def __init__(self, num_threads=8):
        self._nThreads = num_threads

    def extract_save_features(self,
                              files=(),
                              feature_type=None,
                              extract_opts=None,
                              output_dir=None):

        r"""

        Parameters
        ----------
        files : `tuple` of video file paths
        feature_type : `dct` or `aam` are currently supported
        extract_opts : ``dict` holding the configuration for feature extraction
        output_dir

        Returns
        -------

        """

        makedirs(output_dir, exist_ok=True)

        if feature_type == 'dct':
            from .Features import dct
            extractor = dct.DCTFeature(extract_opts,
                                       output_dir)
        elif feature_type == 'pca':
            from .Features import pca
            extractor = pca.PCAFeature(files,
                                       extract_opts,
                                       output_dir)
        elif feature_type == 'landmarks':
            from .Features import landmark
            extractor = landmark.LandmarkFeature(files,
                                                 extract_opts,
                                                 output_dir)

        elif feature_type == 'aam':
            from .Features import aam
            extractor = aam.AAMFeature(extract_opts=extract_opts, output_dir=output_dir)
            extractor.extract_save_features(files)
            return

        elif feature_type == 'aps':
            from .Features import aps
            extractor = aps.APSFeature(files=files, extractOpts=extract_opts, outDir=output_dir)
        else:
            raise Exception('Unknown feature type: ' + feature_type)

        with Pool(self._nThreads) as p:
            p.map(extractor.extract_save_features, files)

    def process_features_write_htk(self,
                                   files=(),
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
        feature_type : `dct`, `landmark`, `aam`
        process_opts : `dict` holding the configuration for feature processing
        frame_rate : `float`
        output_dir : `str`, path to store the extracted features

        Returns
        -------

        """
        makedirs(output_dir, exist_ok=True)
        if feature_type == 'dct':
            from .Features import dct
            processor = dct.DCTFeature(feature_dir=feature_dir)
        elif feature_type == 'pca':
            from .Features import pca
            processor = pca.PCAFeature(vidFiles=(files,),
                                       featDir=feature_dir)
            # extractor.fitData(featOpts)
        elif feature_type == 'aam':
            from .Features import aam
            processor = aam.AAMFeature(process_opts=process_opts)
        else:
            raise Exception('Unknown feature type: ' + feature_type)

        with Pool(self._nThreads) as p:
            p.starmap(_process_one_file,
                      zip(files, repeat(processor), repeat(process_opts), repeat(output_dir), repeat(frame_rate)))


def run(train_files=(),
        test_files=(),
        feature_dir=None,
        hmm_states=None,
        mixtures=None,
        language_model=False,
        config_dir=None,
        experiment_name=None,
        report_results=('train', 'test')
        ):
    r"""

    :param train_files:
    :param test_files:
    :param feature_dir:
    :param hmm_states:
    :param mixtures:
    :param language_model:
    :param config_dir:
    :param experiment_name:
    :param report_results:
    :return:
    """
    htksys = htk.HTKSys(train_files,
                        test_files,
                        feature_dir,
                        hmm_states=hmm_states,
                        mixtures=mixtures,
                        language_model=language_model,
                        config_dir=config_dir,
                        report_results=report_results)
    htksys.run()

    # TODO - wrap the code below in a function

    fld = path.split(path.split(feature_dir)[0])[1]
    fld = './results/' + str(experiment_name) + '/' + fld + '_' + str(hmm_states) + 'states/'
    makedirs(fld, exist_ok=True)

    from glob import glob
    results = glob('./run/results_*')
    from shutil import copy
    for file in results:
        copy(file, fld)


def _write_feature_to_htk(features,
                          out_dir,
                          file,
                          frame_rate):

    r""" Writes to output file the arr array with the given frame rate
    in the HTK binary format
    """

    from struct import pack
    num_frames, feature_size = np.shape(features)

    if path.isabs(file):  # case of AAM
        output_filename = file_to_feature(file, extension='.htk')
    else:  # case of DCT
        output_filename = path.splitext(file)[0] + '.htk'

    outfile = open(out_dir + output_filename, 'wb')

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
    feature_dict = processor.get_feature(file, process_opts)

    for feature_name, feature_value in feature_dict.items():
        makedirs(out_dir + feature_name + '/', exist_ok=True)
        _write_feature_to_htk(feature_value,
                              out_dir + feature_name + '/',
                              file,
                              frame_rate)
