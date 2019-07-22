#!/bin/sh

cd insurance_bot
export PYTHONPATH="."
export ROOT_DIR="root"

#/root/anaconda3/envs/py36/bin/python classification/run_classifier.py

#/root/anaconda3/envs/py36/bin/python classification_2/run_classifier_2.py


#/root/anaconda3/envs/py36/bin/python classification/run_classifier_focal_loss.py



#/root/anaconda3/envs/py36/bin/python log_process/log_eval.py



#/root/anaconda3/envs/py36/bin/python pretrain_LM/finetune_on_pregenerated.py


/root/anaconda3/envs/py36/bin/python classification/run_classifier_tf.py

