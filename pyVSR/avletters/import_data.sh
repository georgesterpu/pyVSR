#!/bin/bash
#this is not guarded against whitespace!
src_dir=/data/corpora/audiovisual/avletters
#import audio data as is with sub directories
mkdir data
ln -s $src_dir/Audio data/
#import video data with name change, dropping "-lips" from "xxx-lips.mat"
mkdir data/Lips
for f in $src_dir/Lips/*.mat; do ln -s $f data/Lips/$(basename ${f/-lips/}); done
#make labels from the file name, this is just the first letter
mkdir data/Label
for f in data/Audio/mfcc/Clean/*.mfcc; do name=$(basename $f ".mfcc"); echo $name ${name/[0-9]_*} > data/Label/$name.mlf; done 
