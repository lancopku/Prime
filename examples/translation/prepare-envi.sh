#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh



echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


BPEROOT=subword-nmt
BPE_TOKENS=10000

URLS=(
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en"
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi"
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en"
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi"
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en"
    "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi"
)
FILES=(
    "train.en"
    "train.vi"
    "tst2012.en"
    "tst2012.vi"
    "tst2013.en"
    "tst2013.vi"
)
src=en
tgt=vi
lang=en-vi
prep=iwslt15.tokenized.en-vi
tmp=$prep/tmp
orig=$prep/orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi
done

cd ../..

TRAIN=$tmp/train.en-vi
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $orig/train.$l >> $TRAIN
done


echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L tst2012.$L tst2013.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $orig/$f > $tmp/bpe.$f
    done
done

for L in $src $tgt; do
    cp $tmp/bpe.train.$L $prep/train.$L
    cp $tmp/bpe.tst2012.$L $prep/valid.$L
    cp $tmp/bpe.tst2013.$L $prep/test.$L
done


TEXT=examples/translation/iwslt15.tokenized.en-vi
python3 preprocess.py --source-lang en --target-lang vi \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt15.tokenized.en-vi.bpe10k \
    --workers 20