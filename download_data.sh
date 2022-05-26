#!/bin/bash

DATA_PATH="./data"

if [[ $1 = "BIN" ]]; then

    curl -L "https://drive.google.com/uc?export=download&id=1TRUoqAtnNTft35NQP2AnHSqxbmKizSi9&confirm=t" -o $DATA_PATH/ModelNet40_Binary.tar
    mkdir -p $DATA_PATH/ModelNet40_Binary
    tar xvf $DATA_PATH/ModelNet40_Binary.tar --directory=$DATA_PATH/ModelNet40_Binary

else
    curl -L "https://drive.google.com/uc?export=download&id=1jEXt5GEemGT3Av4fNGOowtS_0zNr_GTa&confirm=t" -o $DATA_PATH/ModelNet40.tar
    mkdir -p $DATA_PATH/ModelNet40
    tar xvf $DATA_PATH/ModelNet40.tar --directory=$DATA_PATH/ModelNet40
fi

