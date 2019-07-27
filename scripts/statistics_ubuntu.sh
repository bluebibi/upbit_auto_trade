#!/bin/bash
#
cd $HOME/git/upbit_auto_trade

/home/yhhan/anaconda3/envs/upbit_auto_trade/bin/python -m db.statistics >> /home/yhhan/git/upbit_auto_trade/logs/statistics.log 2>&1

