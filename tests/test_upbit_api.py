import unittest
import numpy as np
from common.global_variables import *
from upbit.upbit_api import Upbit
import pprint

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
        pp.pprint(self.upbit.get_current_price(["KRW-BTC", "KRW-XRP"]))

    def test_get_order_book(self):
        # print(get_orderbook(tickers=["KRW-BTC"]))
        pp.pprint(self.upbit.get_orderbook(tickers=["KRW-BTC", "KRW-XRP"]))

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

    def test_get_balance(self):
        pp.pprint(upbit.get_balances())

        # 원화 잔고 조회
        print(upbit.get_balance(ticker="KRW"))
        print(upbit.get_balance(ticker="KRW-BTC"))
        print(upbit.get_balance(ticker="KRW-XRP"))

        # 매도
        # print(upbit.sell_limit_order("KRW-XRP", 1000, 20))

        # 매수
        # print(upbit.buy_limit_order("KRW-XRP", 200, 20))

        # 주문 취소
        # print(upbit.cancel_order('82e211da-21f6-4355-9d76-83e7248e2c0c'))
