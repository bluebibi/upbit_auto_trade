#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.buy >> /Users/yhhan/git/upbit_auto_trade/logs/buy.log 2>&1

