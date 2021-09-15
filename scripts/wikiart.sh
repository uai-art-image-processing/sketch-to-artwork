#!/bin/bash
# Script for the wikiart dataset 
# Source: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

# Get WikiArt dump
wget -c http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
unzip wikiart.zip

# Get WikiArt .csv
cd wikiart
wget -c http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip
unzip wikiart_csv.zip

mkdir ../data/{train,val}
python wikiart.py 