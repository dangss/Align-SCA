#!/bin/sh

python scripts/average_checkpoints.py --inputs $1 --num-epoch-checkpoints 5 --output $1/model.pt
