#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.sell >> /Users/yhhan/git/upbit_auto_trade/logs/sell.log 2>&1

