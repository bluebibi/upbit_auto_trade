#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/home/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m upbit.upbit_recorder >> /home/yhhan/anaconda3/envs/upbit_auto_trade/predict/logs/upbit_record_ubuntu.log 2>&1

#/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.buy

