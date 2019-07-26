#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/home/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.main_prediction >> /home/yhhan/git/upbit_auto_trade/logs/main_prediction_ubuntu.log 2>&1

