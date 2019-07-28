#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

$HOME/anaconda3/envs/upbit_auto_trade/bin/python -m predict.buy >> $HOME/git/upbit_auto_trade/logs/buy.log 2>&1

