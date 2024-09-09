#!/bin/bash

SEGMENT_LEN=0.5
# check if argument is provided
if [ $# -eq 1 ]; then
    SEGMENT_LEN=$1
fi

./src/preprocessor.py ./training_data/raw_audio/youtube_audio/ ./training_data/processed_audio/youtube_processed/ --segment_length $SEGMENT_LEN
