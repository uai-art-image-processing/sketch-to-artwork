#!/bin/bash
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
tar -zxf edges2shoes.tar.gz

find edges2shoes/train -name "*.jpg" > edges2shoes_train.txt
find edges2shoes/val -name "*.jpg" > edges2shoes_val.txt

mv edges2shoes ../data
mv edges2shoes_train.txt ../data
mv edges2shoes_val.txt ../data