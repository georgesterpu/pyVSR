from abc import ABCMeta, abstractmethod


class Feature(metaclass=ABCMeta):
    r"""
    Abstract interface for algorithms that extract features from video files.
    """
    @abstractmethod
    def extract_save_features(self, files):
        r"""
        Pre-computes models or arrays for easier reuse
        Examples: an Active Appearance Model, a DCT matrix
        Parameters
        ----------
        files

        Returns
        -------

        """
        pass

    @abstractmethod
    def get_feature(self, file, feat_opts):
        r"""
        Produces the .htk features files according to the specified processing options,
        while exploiting the pre-computed data.
        Examples: An AAM fitting algorithm uses the pre-computed model to minimise a loss function,
        returning the parameters. A DCT feature processor serializes the DCT matrices and does some postprocessing.

        Parameters
        ----------
        file
        feat_opts

        Returns
        -------

        """
        pass
