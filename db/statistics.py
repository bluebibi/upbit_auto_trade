import smtplib
import sqlite3
import jinja2

from email.mime.text import MIMEText
from common.global_variables import *
from common.utils import *

if os.getcwd().endswith("upbit_auto_trade"):
    pass
elif os.getcwd().endswith("db"):
    os.chdir("..")
else:
    pass

select_all_buy_sell_sql = "SELECT * FROM BUY_SELL ORDER BY id DESC;"


def render_template(**kwargs):
    templateLoader = jinja2.FileSystemLoader(searchpath="db")
    templateEnv = jinja2.Environment(loader=templateLoader)
    templ = templateEnv.get_template("email.html")
    return templ.render(**kwargs)


def buy_sell_tables():
    with sqlite3.connect(sqlite3_db_filename, timeout=10, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()

        rows = cursor.execute(select_all_buy_sell_sql)

        txt = "<tr><th>매수 기준 날짜/시각</th><th>구매 코인</th><th>모델 확신도<br/>(CNN | LSTM)</th><th>구매 가격</th>"
        txt += "<th>현재 가격</th><th>경과 시간</th><th>등락 비율</th><th>상태</th></tr>"
        total_rate = 0.0
        num = 0
        num_success = 0
        num_trail_bought = 0
        num_gain = 0
        num_loss = 0
        for row in rows:
            num += 1
            if row[9] == CoinStatus.success_sold.value:
                num_success += 1
            elif row[9] == CoinStatus.gain_sold.value:
                num_gain += 1
            elif row[9] == CoinStatus.loss_sold.value:
                num_loss += 1
            elif row[9] == CoinStatus.trailed.value:
                num_trail_bought += 1
            elif row[9] == CoinStatus.bought.value:
                num_trail_bought += 1

            total_rate += float(row[8])
            txt += "<tr>"
            txt += "<td>{0}</td><td>{1}</td><td>{2} | {3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}%</td><td>{8}</td>".format(
                row[2],
                row[1],
                convert_unit_2(row[3]),
                convert_unit_2(row[4]),
                row[5],
                row[7],
                elapsed_time_str(row[6], row[2]),
                convert_unit_2(row[8] * 100),
                coin_status_to_hangul(row[9])
            )
            txt += "</tr>"

    return txt, convert_unit_2(total_rate * 100), num, num_trail_bought, num_success, num_gain, num_loss


def main():
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('yh21.han@gmail.com', GOOGLE_APP_PASSWORD)

    buy_sell_text, total_rate, num, num_trail_bought, num_success, num_gain, num_loss = buy_sell_tables()

    html_data = render_template(
        buy_sell_text=buy_sell_text,
        total_rate=total_rate,
        num=num,
        num_trail_bought=num_trail_bought,
        num_success=num_success,
        num_gain=num_gain,
        num_loss=num_loss
    )

    msg = MIMEText(html_data, _subtype="html", _charset="utf-8")
    msg['Subject'] = 'Statistics'

    s.sendmail("yh21.han@gmail.com", "yh21.han@gmail.com", msg.as_string())

    s.quit()


if __name__ =="__main__":
    main()
