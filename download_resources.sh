#!/bin/bash -x

function download_u_shape_transformer_resources () {
    # Model weights
    poetry run gdown https://drive.google.com/uc?id=19a_kDJTT5S96kzwQntEMhSxAPYw4xY2P&confirm=t&uuid=842247f7-f938-4a3b-ad82-b079874e470c
    wait # Wait for poetry process to finish before unziping data

    # Unzip data to destination folder
    unzip saved_models.zip -d divevision/models/UShapeTransformer/

    # Clean up
    rm saved_models.zip
}

function download_cevae_resources () {
    # Metrics parameters
    wget -P divevision/models/CEVAE/metrics/ https://raw.githubusercontent.com/xinntao/EDVR/master/basicsr/metrics/niqe_pris_params.npz
    
    # Model weights
    poetry run gdown -O divevision/models/CEVAE/lsui-cevae-epoch119.ckpt https://drive.google.com/uc?id=1EJHykrwPqegmdWsrAT2Qd8ScYs7flX-i&confirm=t&uuid=197f6233-5f60-4fa9-81ab-af8e2a53d061 &
}

download_u_shape_transformer_resources
download_cevae_resources