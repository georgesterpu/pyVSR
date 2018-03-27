# pyVSR
Python toolkit for Visual Speech Recognition

# About

pyVSR is a Python toolkit aimed at running Visual Speech Recognition (VSR) experiments in a traditional framework (e.g. handcrafted visual features, Hidden Markov Models for pattern recognition).

The main goal of pyVSR is to easily reproduce VSR experiments in order to have a baseline result on most publicly available audio-visual datasets.

# What can you do with pyVSR:

### 1. Fetch a filtered list of files from a dataset
  * currently supported: 

    * TCD-TIMIT
      * speaker-dependent protocol (Gillen)
      * speaker-independent protocol (Gillen)
      * single person

    * OuluVS2
      * speaker-independent protocol (Saitoh)
      * single person

### 2. Extract visual features:
  * Discrete Cosine Transform (DCT)
    * Automatic ROI extraction (grayscale, RGB, DCT)
    * Face alignment (from 5 stable landmarks)
    * Configurable window size
    * Fourth order accurate derivatives
    * Sample rate interpolation
    * Storage in HDF5 format
    
  * Active Appearance Models (AAM)
    * Do NOT require manually annotated landmarks
    * Face, lips, and chin models supported
    * Parameters obtainable either through fitting or projection
    * Implementation based on [Menpo](http://www.menpo.org/)
    
  * Point cloud of facial landmarks 
    * OpenFace wrapper
   
### 3. Train Hidden Markov Models (HMMs)
  * easy HTK wrapper for Python 
  * optional bigram language model
  * multi-threaded support (both for training and decoding at full CPU Power)
  
### 4. Extend the support for additional features
  * pyVSR has a simple, modular, object-oriented architecture

# Examples

Please refer to the attached examples.

pyVSR was re-designed to simplify its usage on multiple datasets.

Users can provide their own dictionaries of (input, output) pairs for all of pyVSR's functionalities.   

# Installing pyVSR
The recommended way is to create an empty `conda` environment and install the following dependencies:

* conda install -c menpo menpo menpofit menpodetect menpowidgets
* conda install -c menpo pango harfbuzz
* conda install h5py
* conda install natsort
* conda install scipy

Alternatively, you can use the `environment.yml` file:
* conda env create -f environment.yml

It is the user's responsibility to compile `OpenFace` and `HTK`.\
Please refer to the documentation upstream:\
[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)\
[HTK 3.5](http://htk.eng.cam.ac.uk/download.shtml)

Add the HTK binaries to the system path (e.g. `/usr/local/bin/`) or to `./pyVSR/bins/htk/`\
Add the OpenFace binaries to `./pyVSR/bins/openface/`

pyVSR was initially developed on a system running Manjaro Linux, frequently updated from the `testing` repositories.
We also succesfully tested it on Windows systems.

If you are not interested in using the AAM module, you can skip installing a great amount of Python packages.
We recommend running the example scripts and installing the missing dependencies (opencv, dlib, numpy).

# How to cite
If you use this work, please cite it as:

George Sterpu and Naomi Harte.
__Towards lipreading sentences using active appearance models.__
In AVSP, Stockholm, Sweden, August 2017.

[Bib](https://raw.githubusercontent.com/georgesterpu/georgesterpu.github.io/master/bibs/avsp2017.bib)
# Contact
We are always happy to hear from you:

George Sterpu sterpug [at] tcd.ie \
Naomi Harte nharte [at] tcd.ie
