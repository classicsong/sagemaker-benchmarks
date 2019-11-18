#!/usr/bin/env bash
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# PreRequisities
# Install mxnet >=1.6 and dgl >= 0.4.0

echo "I'm a sm train bash"

pip3 install tqdm
# Clone source to get launch.py script to start training job
git clone https://github.com/dmlc/dgl.git

# Data will auto-downloaded by the program
# Example command to start the training job
# Specify hosts in the file `hosts`

pushd dgl/apps/kg/

python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv
