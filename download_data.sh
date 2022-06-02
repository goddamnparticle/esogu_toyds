#!/bin/bash

DATA_PATH="./data"

if [[ $1 = "BIN" ]]; then
    curl -L "https://drive.google.com/uc?export=download&id=14Bgxb7ft4QJy97KhIQ0yYHk_7yQgE-Nr&confirm=t" -o $DATA_PATH/ESOGU_ToyDS_S4096AB.tar
    mkdir -p $DATA_PATH/ESOGU_ToyDS_S4096AB
    tar xvf $DATA_PATH/ESOGU_ToyDS_S4096AB.tar --directory=$DATA_PATH/ESOGU_ToyDS_S4096AB
else
    curl -L "https://drive.google.com/uc?export=download&id=1ChN4KUpgYlVHFm2ElTTY30uvx0u8pM4a&confirm=t" -o $DATA_PATH/ESOGU_ToyDS_A.tar
    mkdir -p $DATA_PATH/ESOGU_ToyDS_A
    tar xvf $DATA_PATH/ESOGU_ToyDS_A.tar --directory=$DATA_PATH/ESOGU_ToyDS_A
fi

