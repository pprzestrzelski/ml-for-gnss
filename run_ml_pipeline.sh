#!/bin/bash

TRAINING_SCRIPT=scripts/desperation/train_multiple_networks.py
PREDICTION_SCRIPT=scripts/desperation/predict_multiple_clocks.py

CLOCK_DATA_FLODER=local/clock_data
NETWORK_FOLDER=local/nets
PREDICTION_FOLDER=local/predictions

BIAS_COLUMN_NAME=Clock_bias
INPUT_SIZE=32
EPOCH_COUNT=20
TRAINING_TO_VALIDATION_COEFFICENT=0.8
SCALING_FACTOR=0.0800576415016
PREDICTION_DEPTH=10
PREDICTION_ZERO_EPOCH=2011.0
PREDICTION_EPOCH_STEP=0.011

init_dirs() {
    if [ ! -d "$CLOCK_DATA_FLODER" ]; then
        echo "Directory $CLOCK_DATA_FLODER does not exist; creating..."
        echo "   !!! Add data to feed neural networks !!!"
        mkdir -p $CLOCK_DATA_FLODER
    fi

    if [ ! -d "$NETWORK_FOLDER" ]; then
        echo "Directory $NETWORK_FOLDER does not exist; creating..."
        echo "   !!! Remember to perform training of networks !!!"
        mkdir -p $NETWORK_FOLDER
    fi

    if [ ! -d "$PREDICTION_FOLDER" ]; then
        echo "Directory $PREDICTION_FOLDER does not exist; creating..."
        mkdir -p $PREDICTION_FOLDER
    fi
}

init_dirs

read -n1 -p "Execute training? [y,n]" EXECUTE_TRAINING
if [[ $EXECUTE_TRAINING == 'y' ]]; then
    python $TRAINING_SCRIPT $CLOCK_DATA_FLODER $BIAS_COLUMN_NAME $INPUT_SIZE $EPOCH_COUNT $TRAINING_TO_VALIDATION_COEFFICENT $NETWORK_FOLDER $SCALING_FACTOR
fi
python $PREDICTION_SCRIPT $CLOCK_DATA_FLODER $BIAS_COLUMN_NAME $NETWORK_FOLDER $INPUT_SIZE $PREDICTION_DEPTH $SCALING_FACTOR $PREDICTION_FOLDER $PREDICTION_ZERO_EPOCH $PREDICTION_EPOCH_STEP
