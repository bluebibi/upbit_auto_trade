#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/home/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m predict.make_models >> /home/yhhan/git/upbit_auto_trade/logs/make_models.log 2>&1
