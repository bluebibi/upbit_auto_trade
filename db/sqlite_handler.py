import sqlite3


class SqliteHandler:
    def __init__(self, sqlite3_db_filename):
        self.conn = sqlite3.connect(sqlite3_db_filename, timeout=10, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout = 3000")

    def create_tables(self, coin_names):
        self.cursor = self.conn.cursor()
        for coin_name in coin_names:
            ticker = "KRW_" + coin_name
            self.cursor.execute("CREATE TABLE IF NOT EXISTS {0} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                "datetime TEXT, open_price FLOAT, high_price FLOAT, low_price FLOAT, close_price FLOAT, volume FLOAT, "
                                "total_ask_size FLOAT, total_bid_size FLOAT, btmi FLOAT, btmi_rate FLOAT, "
                                "btai FLOAT, btai_rate FLOAT)".format(ticker))

        self.conn.commit()

    def drop_tables(self, coin_names):
        self.cursor = self.conn.cursor()
        for coin_name in coin_names:
            ticker = "KRW_" + coin_name
            self.cursor.execute("DROP TABLE IF EXISTS {0}".format(ticker))

        self.conn.commit()

    def close(self):
        self.conn.close()


