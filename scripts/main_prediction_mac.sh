#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.main_prediction >> /Users/yhhan/git/upbit_auto_trade/predict/logs/main_prediction_ubuntu.log 2>&1

