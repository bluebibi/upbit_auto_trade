#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m upbit.upbit_recorder >> /Users/yhhan/git/upbit_auto_trade/logs/upbit_record.log 2>&1
#/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.buy

