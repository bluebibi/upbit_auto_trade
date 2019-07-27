#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/home/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.sell >> /home/yhhan/git/upbit_auto_trade/logs/sell.log 2>&1

