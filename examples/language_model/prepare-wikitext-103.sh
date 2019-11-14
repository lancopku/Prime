#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
URLS="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
FILES="wikitext-103-v1.zip"
for i in 1
do
    file=${FILES}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done
cd ..
