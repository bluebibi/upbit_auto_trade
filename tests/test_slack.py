import unittest
from upbit.slack import PushSlack


class TestSlack(unittest.TestCase):
    def setUp(self):
        self.pusher = PushSlack()

    def test_send_message(self):
        self.pusher.send_message("me", "<https://www.bithumb.com/trade/order/PPT|This> is the 테스트 msg")

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
