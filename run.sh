#!/bin/sh

cd insurance_bot
export PYTHONPATH="."
export ROOT_DIR="root"

/root/anaconda3/envs/py36/bin/python classification/run_classifier.py

#/root/anaconda3/envs/py36/bin/python classification_2/run_classifier_2.py


