#!/bin/bash
# Script for the wikiart dataset 
# Source: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset
mkdir -p data
cd data

# Get WikiArt dump
wget -c http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip 
unzip wikiart.zip

# Get WikiArt .csv
cd wikiart
wget -c http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart_csv.zip
unzip wikiart_csv.zip

# Prepare train
mkdir -p {train,test,val}
python ../../scripts/wikiart.py 

# Process the edge images
mkdir -p edges/{train,test,val}
python ../scripts/wikiart-edges.py -s train -d edges/train 
python ../scripts/wikiart-edges.py -s test -d edges/test 
python ../scripts/wikiart-edges.py -s val -d edges/val 

cd ..

find $(pwd)/wikiart/train -name "*.jpg" > wikiart_train.txt
find $(pwd)/wikiart/test -name "*.jpg" > wikiart_test.txt
find $(pwd)/wikiart/val -name "*.jpg" > wikiart_val.txt

find $(pwd)/wikiart/edges/train -name "*.jpg" > wikiart_edges_train.txt
find $(pwd)/wikiart/edges/test -name "*.jpg" > wikiart_edges_test.txt
find $(pwd)/wikiart/edges/val -name "*.jpg" > wikiart_edges_val.txt

echo "Done"
exit 0