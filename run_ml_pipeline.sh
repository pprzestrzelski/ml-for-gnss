#!/bin/bash

TRAINING_SCRIPT=scripts/desperation/train_multiple_networks.py
PREDICTION_SCRIPT=scripts/desperation/predict_multiple_clocks.py

TRAINING_DATA_FOLDER=local/training
PREDICTION_INPUT_FOLDER=local/prediction_input
NETWORK_FOLDER=local/networks
PREDICTION_FOLDER=local/lstm_predicted

BIAS_COLUMN_NAME=Clock_bias
INPUT_SIZE=32
EPOCH_COUNT=20
TRAINING_TO_VALIDATION_COEFFICENT=0.8
SCALING_FACTOR=0.0800576415016
PREDICTION_DEPTH=95
PREDICTION_ZERO_EPOCH=2010.0
PREDICTION_EPOCH_STEP=0.001488095238

init_dirs() {
    if [ ! -d "$TRAINING_DATA_FOLDER" ]; then
        echo "Directory $TRAINING_DATA_FOLDER does not exist; creating..."
        echo "   !!! Add data to feed neural networks !!!"
        mkdir -p $TRAINING_DATA_FOLDER
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
    python $TRAINING_SCRIPT $TRAINING_DATA_FOLDER $BIAS_COLUMN_NAME $INPUT_SIZE $EPOCH_COUNT $TRAINING_TO_VALIDATION_COEFFICENT $NETWORK_FOLDER $SCALING_FACTOR
fi

read -n1 -p "Generate predictions? [y,n]" GENERATE_PREDICTIONS
if [[ $GENERATE_PREDICTIONS == 'y' ]]; then
python $PREDICTION_SCRIPT $PREDICTION_INPUT_FOLDER $BIAS_COLUMN_NAME $NETWORK_FOLDER $INPUT_SIZE $PREDICTION_DEPTH $SCALING_FACTOR $PREDICTION_FOLDER $PREDICTION_ZERO_EPOCH $PREDICTION_EPOCH_STEP
fi


