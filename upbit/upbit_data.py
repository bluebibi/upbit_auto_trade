from common.global_variables import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

select_all = "SELECT * FROM {0};"
select_recent_window = "SELECT * FROM {0} ORDER BY id DESC LIMIT {1};"


class UpbitData:
    def __init__(self, coin_name):
        self.coin_name = coin_name

    def get_buy_for_data(self):
        df = pd.read_sql_query(
            select_recent_window.format("KRW_" + self.coin_name, WINDOW_SIZE),
            SQL_HANDLER.conn
        )

        df = df.sort_values('id', ascending=True)

        if TRAIN_COLS_FULL:
            df = df.drop(["id", "datetime"], axis=1)
        else:
            df = df.drop(["id", "datetime", "total_ask_size", "total_bid_size", "btmi", "btmi_rate", "btai", "btai_rate"], axis=1)

        min_max_scaler = MinMaxScaler()
        x_normalized = min_max_scaler.fit_transform(df.values)
        x_normalized = torch.from_numpy(x_normalized).float().to(DEVICE)

        if USE_CNN_MODEL:
            return x_normalized.unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            return x_normalized.unsqueeze(dim=0)

    def get_data(self):
        df = pd.read_sql_query(select_all.format("KRW_" + self.coin_name), SQL_HANDLER.conn)
        if TRAIN_COLS_FULL:
            df = df.drop(["id", "datetime"], axis=1)
        else:
            df = df.drop(["id", "datetime", "total_ask_size", "total_bid_size", "btmi", "btmi_rate", "btai", "btai_rate"], axis=1)

        data = torch.from_numpy(df.values).to(DEVICE)

        min_max_scaler = MinMaxScaler()
        data_normalized = min_max_scaler.fit_transform(data)
        data_normalized = torch.from_numpy(data_normalized).to(DEVICE)

        x, x_normalized, y, y_normalized, y_up, one_rate, total_size = self.build_timeseries(
            data=data,
            data_normalized=data_normalized,
            window_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE
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

        if USE_CNN_MODEL:
            return x_train.unsqueeze(dim=1), x_train_normalized.unsqueeze(dim=1), y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid.unsqueeze(dim=1), x_valid_normalized.unsqueeze(dim=1), y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size
        else:
            return x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
                   x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size

    @staticmethod
    def build_timeseries(data, data_normalized, window_size, future_target_size, up_rate):
        y_col_index = 3
        future_target = future_target_size - 1

        dim_0 = data.shape[0] - window_size - future_target
        dim_1 = data.shape[1]

        x = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)
        x_normalized = torch.zeros((dim_0, window_size, dim_1)).to(DEVICE)
        y = torch.zeros((dim_0,)).to(DEVICE)
        y_normalized = torch.zeros((dim_0,)).to(DEVICE)
        y_up = torch.zeros((dim_0,)).float().to(DEVICE)

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


def main():
    upbit_data = UpbitData('BTC')

    x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size,\
    x_test, x_test_normalized, y_test, y_test_normalized, y_up_test, one_rate_test, test_size = upbit_data.get_data(
        windows_size=WINDOW_SIZE,
        future_target_size=FUTURE_TARGET_SIZE,
        up_rate=UP_RATE,
        cnn=True,
    )

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


if __name__ == "__main__":
    main()
