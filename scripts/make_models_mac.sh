#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/Users/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.make_models >> /Users/yhhan/git/upbit_auto_trade/logs/make_models.log 2>&1

