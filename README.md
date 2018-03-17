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

## The examples shown here or in the python scripts are no longer functional and maintained, please check the documentation of each Class for the updated API'

##### 1. Loading dataset files
```python
from pyVSR import tcdtimit

dataset_dir = '/path/to/dataset/tcdtimit/'

train, test = tcdtimit.files.request_files(
    dataset_dir=dataset_dir,
    protocol='single_volunteer',
    speaker_id='24M')
```

##### 2. Extract DCT Features
First store the full ROI sequences:
```python
import pyVSR
experiment = pyVSR.AVSR(num_threads=4)  

experiment.extract_save_features(
        files=train+test,
        feature_type='roi',
        extract_opts={
            'align': True,
            'gpu': True,
            'color': False,
            'border': 15,
            'window_size': (36, 36),
        },
        output_dir='./run/features/roi_gray/'
    )
```

Then compute the DCT coefficients on the fly and write .htk binary files:

```python
features_train = files_to_features(train, extension='.h5')
features_test = files_to_features(test, extension='.h5')

experiment.process_features_write_htk(
        files=dct_train + dct_test,
        feature_dir='./run/features/roi_gray/',
        feature_type='dct',
        process_opts={
            'compute_dct': True,
            'mask': '1-44',
            'keep_basis': True,
            'delta': True,
            'double_delta': True,
            'derivative_order': 'first',
            'interpolation_factor': 1},
        frame_rate=30,
        output_dir='./run/features/htk_dct_gray/'
    )
```

##### 3. Get facial landmarks estimates
```python
import pyVSR
experiment = pyVSR.AVSR(num_threads=2)

experiment.extract_save_features(
    files=train + test,
    feature_type='landmarks',
    extract_opts=None,
    output_dir='./run/features/facial_landmarks/'
)
```

##### 4. Train an Active Appearance Model (AAM)
```python
experiment.extract_save_features(
    files=train[::14],
    feature_type='aam',
    extract_opts={
       'warp':'patch',
       'resolution_scales': (0.25, 0.5, 1.0),
       'patch_shape':((5,5), (10,10), (17,17)),
       'max_shape_components':20,
       'max_appearance_components': 150,
       'diagonal': 150,
       'features': 'no_op',
       'landmark_dir': './run/features/facial_landmarks/',
       'landmark_group': 'pts_face',
       'confidence_thresh':0.94,
       'kept_frames': 0.03,
       'greyscale':False,
       'model_name': 'face_hnop_34M.pkl'},
    output_dir='./run/features/aam/'
)
```

##### 5. Extract AAM Features
```python
experiment.process_features_write_htk(
    files=test,
    feature_dir='./pyVSR/pretrained/',
    feature_type='aam',
    process_opts={
        'face_detector': 'dlib',
        'landmark_fitter': 'aam',
        'aam_fitter': './run/features/aam/face_hnop_34M.pkl',
        'parameters_from': 'lk_fitting',
        'projection_aam': None,
        'shape': 'face',
        'part_aam': None,
        'confidence_thresh': 0.84,
        'shape_components': [10, 15, 20],
        'appearance_components': [20, 30, 150],
        'max_iters': [10, 10, 5],
        'landmark_dir': './run/features/facial_landmarks/',
        'log_errors': True,
        'log_title': '34M/log_demo'},
    out_dir='./run/features/htk_aam/'
)
```

##### 6. Train a HMM, store decoding results

```python
train_feat = pyVSR.utils.files_to_features(train, extension='.htk')
test_feat = pyVSR.utils.files_to_features(test, extension='.htk')

pyVSR.run(
    train_files=train_feat,
    test_files=test_feat,
    feature_dir='./run/features/htk_dct/',
    hmm_states=3,
    mixtures=(2, 3, 5, 7, 9, 11, 14, 17, 20),
    language_model=False,
    config_dir='./pyVSR/tcdtimit/htkconfigs/',
    report_results=('train', 'test'),
    experiment_name='dct_24M'
    )
```


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
We will be testing the code soon on other platforms.

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
