#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <zip file path> <output folder>"
    exit 1
fi

zip_file=$1
output_folder=$2

# Check if the zip file exists
if [ ! -f $zip_file ]; then
    echo "Error: $zip_file does not exist"
    exit 1
fi

# create the output folder
mkdir -p $output_folder

# Unzip the file
unzip $zip_file -d $output_folder

cd $output_folder

cd Replica_Dataset

# Unzip the scene files
for scene in $(ls *.zip); do
    unzip $scene
    rm $scene
done

# Unzip instance segmentation
cd Replica_Instance_Segmentation
for scene in $(ls ); do
    cd $scene
    cd Sequence_1
    unzip semantic_instance.zip
    rm semantic_instance.zip
    cd ../..
done