from common.config import *
from db.sqlite_handler import SqliteHandler
from upbit.upbit_api import Upbit
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

select_by_datetime = "SELECT * FROM {0};"


class Upbit_Data:
    def __init__(self, coin_name, train_cols):
        self.sql_handler = SqliteHandler(sqlite3_db_filename)
        self.upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT)
        self.coin_name = coin_name
        self.train_cols = train_cols

    def get_data(self, num, coin_name, windows_size=10, future_target_size=6, up_rate=0.05, cnn=False, verbose=True):
        df = pd.read_sql_query(select_by_datetime.format("KRW_" + self.coin_name), self.sql_handler.conn)
        df = df.drop(["id", "datetime"], axis=1)

        data = torch.from_numpy(df.values).to(device)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(data)
        data_normalized = torch.from_numpy(data_normalized).to(device)

        x, x_normalized, y, y_normalized, y_up, one_rate, total_size = self.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=windows_size,
            future_target_size=future_target_size,
            up_rate=up_rate
        )

        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = list(set(indices[:int(total_size * 0.8)]))
        validation_indices = list(set(range(total_size)) - set(train_indices))

        x_train = x[train_indices]
        x_train_normalized = x_normalized[train_indices]

        x_valid = x[validation_indices]
        x_valid_normalized = x_normalized[validation_indices]

        y_train = y[train_indices]
        y_train_normalized = y_normalized[train_indices]

        y_valid = y[validation_indices]
        y_valid_normalized = y_normalized[validation_indices]

        y_up_train = y_up[train_indices]
        y_up_valid = y_up[validation_indices]

        one_rate_train = y_up_train.sum().float() / y_up_train.size(0)
        one_rate_valid = y_up_valid.sum().float() / y_up_valid.size(0)

        train_size = x_train.size(0)
        valid_size = x_valid.size(0)

        if cnn:
            return x_train.unsqueeze(dim=1), x_train_normalized.unsqueeze(dim=1), y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid.unsqueeze(dim=1), x_valid_normalized.unsqueeze(dim=1), y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size
        else:
            return x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size

    def build_timeseries(self, data, data_normalized, window_size, future_target_size, up_rate):
        y_col_index = 3
        future_target = future_target_size - 1

        dim_0 = data.shape[0] - window_size - future_target
        dim_1 = data.shape[1]

        x = torch.zeros((dim_0, window_size, dim_1)).to(device)
        x_normalized = torch.zeros((dim_0, window_size, dim_1)).to(device)
        y = torch.zeros((dim_0,)).to(device)
        y_normalized = torch.zeros((dim_0,)).to(device)
        y_up = torch.zeros((dim_0,)).float().to(device)

        for i in range(dim_0):
            x[i] = data[i: i + window_size]
            x_normalized[i] = data_normalized[i: i + window_size]

        count_one = 0
        for i in range(dim_0):
            max_price = -1.0
            max_price_normalized = -1.0

            for j in range(future_target + 1):
                future_price = data[i + window_size + j, y_col_index]
                future_price_normalized = data_normalized[i + window_size + j, y_col_index]

                if future_price > max_price:
                    max_price = future_price
                    max_price_normalized = future_price_normalized

            y[i] = max_price
            y_normalized[i] = max_price_normalized

            if y[i] > x[i][-1, y_col_index] * (1 + up_rate):
                y_up[i] = 1
                count_one += 1

        return x, x_normalized, y, y_normalized, y_up, count_one / dim_0, dim_0


def get_data_loader(x, x_normalized, y, y_normalized, y_up_train, batch_size, suffle=True):
    total_size = x.size(0)
    if total_size % batch_size == 0:
        num_batches = int(total_size / batch_size)
    else:
        num_batches = int(total_size / batch_size) + 1

    for i in range(num_batches):
        if suffle:
            indices = np.random.choice(total_size, batch_size)
        else:
            indices = np.asarray(range(i * batch_size, min((i + 1) * batch_size, total_size)))

        yield x[indices], x_normalized[indices], y[indices], y_normalized[indices], y_up_train[indices], num_batches


if __name__ == "__main__":
    upbit_data = Upbit_Data('BTC', TRAIN_COLS)

    x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
    x_test, x_test_normalized, y_test, y_test_normalized, y_up_test, one_rate_test, test_size = upbit_data.get_data(
        num=1,
        coin_name='BTC',
        windows_size=WINDOW_SIZE,
        future_target_size=FUTURE_TARGET_SIZE,
        up_rate=UP_RATE,
        cnn=True,
        verbose=VERBOSE
    ) # 과거 3시간 데이터(6포인트)를 기반으로 현재 기준 향후 12시간 이내 2% 상승 예측

    print(x_train.shape)
    print(x_train_normalized.shape)
    print(y_train.shape)
    print(y_train_normalized.shape)
    print(y_up_train.shape)
    print(x_train[32])
    print(y_train[32])
    print(y_up_train[32])

    print(y_up_train)
    print()

    print(x_test.shape)
    print(x_test_normalized.shape)
    print(y_test.shape)
    print(y_test_normalized.shape)
    print(y_up_test.shape)

    batch_size = 4
    train_data_loader = get_data_loader(
        x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, batch_size=4, suffle=False
    )

    test_data_loader = get_data_loader(
        x_test, x_test_normalized, y_test, y_test_normalized, y_up_test, batch_size=4, suffle=False
    )

    print()
    for i, (x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, num_batches) in enumerate(train_data_loader):
        print(i)
        print(x_train.shape)
        print(x_train_normalized.shape)
        print(y_train.shape)
        print(y_train_normalized.shape)
        print(y_up_train.shape)
        print(y_up_train)

    print()
    for i, (x_test, x_test_normalized, y_test, y_test_normalized, y_up_test, num_batches) in enumerate(test_data_loader):
        print(i)
        print(x_test.shape)
        print(x_test_normalized.shape)
        print(y_test.shape)
        print(y_test_normalized.shape)
        print(y_up_test.shape)
        print(y_up_test)