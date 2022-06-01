#!/bin/bash

DATA_PATH="./data"

if [[ $1 = "BIN" ]]; then
    curl -L "https://drive.google.com/uc?export=download&id=1WisS_DvtqVD4T3mxI2UcsxplMjLLy6eW&confirm=t" -o $DATA_PATH/ESOGU_ToyDS_S4096NAB.tar
    mkdir -p $DATA_PATH/ESOGU_ToyDS_S4096NAB
    tar xvf $DATA_PATH/ESOGU_ToyDS_S4096NAB.tar --directory=$DATA_PATH/ESOGU_ToyDS_S4096NAB
else
    curl -L "https://drive.google.com/uc?export=download&id=1_98r2c75YWbcJ9YK3JJ3Oe_IvYgn5RY5&confirm=t" -o $DATA_PATH/ESOGU_ToyDS_NA.tar
    mkdir -p $DATA_PATH/ESOGU_ToyDS_NA
    tar xvf $DATA_PATH/ESOGU_ToyDS_NA.tar --directory=$DATA_PATH/ESOGU_ToyDS_NA
fi

