from common.global_variables import *


select_all_bought_coin_names_sql = "SELECT * FROM BUY_SELL WHERE status=?;"
insert_buy_try_coin_info = "UPDATE BUY_SELL SET trail_datetime=?, trail_price=?, trail_rate=? WHERE coin_name=? and buy_datetime=?;"


def select_all_bought_coin_names():
    cursor = SQL_HANDLER.conn.cursor()
    cursor.execute(select_all_bought_coin_names_sql, (CoinStatus.bought.value,))

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    SQL_HANDLER.conn.commit()


def main():
    select_all_bought_coin_names()


if __name__ == "__main__":
    main()