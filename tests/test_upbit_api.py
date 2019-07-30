import unittest
import numpy as np
from pytz import timezone

from common.global_variables import *
from upbit.upbit_api import Upbit
import pprint
import datetime as dt

pp = pprint.PrettyPrinter(indent=2)


class UpBitAPITestCase(unittest.TestCase):
    def setUp(self):
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, fmt)

    def test_get_tickers(self):
        pp.pprint(self.upbit.get_tickers(fiat="KRW"))

    def test_get_ohlcv(self):
        # print(get_ohlcv("KRW-BTC"))
        # print(get_ohlcv("KRW-BTC", interval="day", count=5))
        # print(get_ohlcv("KRW-BTC", interval="minute1"))
        # print(get_ohlcv("KRW-BTC", interval="minute3"))
        # print(get_ohlcv("KRW-BTC", interval="minute5"))
        pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute10"))
        # print(get_ohlcv("KRW-BTC", interval="minute15"))
        # pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute30"))
        # print(get_ohlcv("KRW-BTC", interval="minute60"))
        # print(get_ohlcv("KRW-BTC", interval="minute240"))
        # print(get_ohlcv("KRW-BTC", interval="week"))
        # pp.pprint(self.upbit.get_daily_ohlcv_from_base("KRW-BTC", base=9))
        # print(get_ohlcv("KRW-BTC", interval="day", count=5))

    def test_get_current_price(self):
        # print(get_current_price("KRW-BTC"))
        pp.pprint(self.upbit.get_current_price(
            ['KRW-GAS', 'KRW-MOC', 'KRW-IQ', 'KRW-WAX', 'KRW-NEO', 'KRW-AERGO', 'KRW-MEDX', 'KRW-XMR',
             'KRW-OST', 'KRW-STRAT', 'KRW-IOST', 'KRW-ONT', 'KRW-BSV']))

    def test_get_order_book(self):
        now = dt.datetime.now(timezone('Asia/Seoul'))
        now_str = now.strftime(fmt)
        current_time_str = now_str.replace("T", " ")
        current_time_str = current_time_str[:-3] + ":00"

        order_book = self.upbit.get_orderbook(tickers="KRW-BTC")

        order_book_units = order_book["orderbook_units"]
        ask_price_lst = []
        ask_size_lst = []
        bid_price_lst = []
        bid_size_lst = []
        for item in order_book_units:
            ask_price_lst.append(item["ask_price"])
            ask_size_lst.append(item["ask_size"])
            bid_price_lst.append(item["bid_price"])
            bid_size_lst.append(item["bid_size"])

        timestamp = order_book['timestamp']
        total_ask_size = order_book['total_ask_size']
        total_bid_size = order_book['total_bid_size']


        print(current_time_str)
        print(order_book)

    def test_get_market_index(self):
        pp.pprint(self.upbit.get_market_index())

    def test_get_all_coin_names(self):
        coin_names = self.upbit.get_all_coin_names()
        print(coin_names)
        print(len(coin_names))

    def test_get_coin_info(self):
        pp.pprint(self.upbit.get_ohlcv("KRW-BTC", interval="minute10"))
        pp.pprint(self.upbit.get_orderbook(tickers="KRW-BTC"))
        pp.pprint(self.upbit.get_market_index())
        pass

    def test_get_expected_buy_coin_price_for_krw(self):
        expected_price = self.upbit.get_expected_buy_coin_price_for_krw("KRW-OMG", 1000000, TRANSACTION_FEE_RATE)
        print(expected_price)

    def test_get_expected_sell_coin_price_for_volume(self):
        expected_price = self.upbit.get_expected_sell_coin_price_for_volume("KRW-OMG", 548.7964360338357, TRANSACTION_FEE_RATE)
        print(expected_price)

    def test_get_balance(self):
        pp.pprint(self.upbit.get_balances())

        # 원화 잔고 조회
        print(self.upbit.get_balance(ticker="KRW"))
        print(self.upbit.get_balance(ticker="KRW-BTC"))
        print(self.upbit.get_balance(ticker="KRW-XRP"))

        # 매도
        # print(upbit.sell_limit_order("KRW-XRP", 1000, 20))

        # 매수
        # print(upbit.buy_limit_order("KRW-XRP", 200, 20))

        # 주문 취소
        # print(upbit.cancel_order('82e211da-21f6-4355-9d76-83e7248e2c0c'))
