#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi

CORPUS=text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
VERBOSE=2
MEMORY=4.0

VOCAB_MIN_COUNT=1

SYMMETRIC=1
WINDOW_SIZE=3

VECTOR_SIZE=20
NUM_THREADS=8
MAX_ITER=15
ETA=0.06
BINARY=2
MODEL=2
X_MAX=10
ALPHA=0.55
FINAL_OUTPUT_FILE=vectors

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi
echo

the_cmd="$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
echo "$ $the_cmd"
eval "$the_cmd"

the_cmd="$BUILDDIR/cooccur -symmetric $SYMMETRIC -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
echo "$ $the_cmd"
eval "$the_cmd"

the_cmd="$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
echo "$ $the_cmd"
eval "$the_cmd"

the_cmd="$BUILDDIR/glove -save-file $FINAL_OUTPUT_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -eta $ETA -alpha $ALPHA -vector-size $VECTOR_SIZE -model $MODEL -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
echo "$ $the_cmd"
eval "$the_cmd"

if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ $PYTHON eval/python/evaluate.py"
       $PYTHON eval/python/evaluate.py
   fi
fi
