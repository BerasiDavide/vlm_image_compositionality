#!/bin/sh

# download the data 
wget -O data/waterbirds.tar.gz https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz

# extract the data
tar -C data -xf data/waterbirds.tar.gz

# remove the tar file
rm data/waterbirds.tar.gz

# reorganize data
python3 datasets/reorganize_waterbirds.py

echo "New dataset created: data/waterbirds"
