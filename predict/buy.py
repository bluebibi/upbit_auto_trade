import glob
import os
import time
from common.global_variables import *
from predict.model_cnn import CNN
from predict.model_rnn import LSTM
from upbit.upbit_data import UpbitData
from pytz import timezone
from datetime import datetime


select_by_datetime = "SELECT * FROM {0} WHERE datetime='{1}';"


def get_good_quality_coin_names_for_buy():
    models = {}
    files = glob.glob('./models/*')
    for f in files:
        if os.path.isfile(f):
            coin_name = f.split("_")[0].replace("./models/", "")

            if USE_CNN_MODEL:
                model = CNN(input_width=INPUT_SIZE, input_height=WINDOW_SIZE).to(DEVICE)
            else:
                model = LSTM(input_size=INPUT_SIZE).to(DEVICE)
            model.load_state_dict(torch.load(f))
            model.eval()

            models[coin_name] = model

    return models


def get_db_right_time_coin_names():
    coin_names = {}
    now = datetime.now(timezone('Asia/Seoul'))
    now_str = now.strftime(fmt)
    current_time_str = now_str.replace("T", " ")
    current_time_str = current_time_str[:-4] + "0:00"

    cursor = SQL_HANDLER.conn.cursor()
    all_coin_names = UPBIT.get_all_coin_names()
    for coin_name in all_coin_names:
        cursor.execute(select_by_datetime.format("KRW_" + coin_name, current_time_str))
        row = cursor.fetchall()
        if len(row) == 1:
            coin_names[coin_name] = current_time_str
    return coin_names


def evaluate_coin_by_models(model, coin_name):
    upbit_data = UpbitData(coin_name)
    x = upbit_data.get_buy_for_data()

    out = model.forward(x)
    out = torch.sigmoid(out)
    t = torch.Tensor([0.5])
    output_index = (out > t).float() * 1

    print(out.item(), int(output_index.item()))


def main():
    good_models = get_good_quality_coin_names_for_buy()
    right_time_coin_names = get_db_right_time_coin_names()

    for coin_name in right_time_coin_names:
        if coin_name in good_models:
            print(coin_name, right_time_coin_names[coin_name])
            evaluate_coin_by_models(good_models[coin_name], coin_name)


if __name__ == "__main__":
    main()
