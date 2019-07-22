# https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb
import glob

import torch
import torch.nn as nn

from conf.config import CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT, NUM_EPOCHS
from conf.config import WINDOW_SIZE, FUTURE_TARGET_SIZE, UP_RATE
from upbit.upbit_api import Upbit
from upbit.upbit_data import Upbit_Data, get_data_loader
import matplotlib.pyplot as plt

from predict.rnn_model import LSTM
from predict.cnn_model import CNN
from predict.early_stopping import EarlyStopping
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists("./models/"):
    os.makedirs("./models/")

files = glob.glob('./models/*')
for f in files:
    os.remove(f)

if not os.path.exists("./graphs/"):
    os.makedirs("./graphs/")

files = glob.glob('./graphs/*')
for f in files:
    os.remove(f)


def save_graph(coin_name, val_loss_min, last_save_epoch, valid_size, one_count_rate, avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list):
    plt.clf()

    fig = plt.figure()  # an empty figure with no axes
    fig.suptitle('{0} - Loss and Accuracy'.format(coin_name))  # Add a title so we know which it is

    fig, ax_lst = plt.subplots(2, 2, gridspec_kw={'hspace': 0.35})
    fig.tight_layout()

    ax_lst[0][0].plot(range(len(avg_train_losses)), avg_train_losses)
    ax_lst[0][0].set_title('AVG. TRAIN LOSSES', fontweight="bold", size=10)

    ax_lst[0][1].plot(range(len(train_accuracy_list)), train_accuracy_list)
    ax_lst[0][1].set_title('TRAIN ACCURACY CHANGE', fontweight="bold", size=10)
    ax_lst[1][1].set_xlabel('EPISODES', size=10)

    ax_lst[1][0].plot(range(len(avg_valid_losses)), avg_valid_losses)
    ax_lst[1][0].set_title('AVG. VALIDATION LOSSES', fontweight="bold", size=10)

    ax_lst[1][1].plot(range(len(valid_accuracy_list)), valid_accuracy_list)
    ax_lst[1][1].set_title('VALIDATION ACCURACY CHANGE', fontweight="bold", size=10)
    ax_lst[1][1].set_xlabel('EPISODES', size=10)

    plt.savefig("./graphs/{0}_{1:.4f}_{2}_{3}_{4:.4f}.png".format(coin_name, val_loss_min, last_save_epoch, valid_size, one_count_rate))
    plt.close('all')


if __name__ == "__main__":
    train_cols = ["open_price", "high_price", "low_price", "close_price", "volume", "total_ask_size",
                      "total_bid_size", "btmi", "btmi_rate", "btai", "btai_rate"]
    # train_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]

    input_size = len(train_cols)
    batch_size = 6
    lr = 0.001

    upbit = Upbit(CLIENT_ID_UPBIT, CLIENT_SECRET_UPBIT)
    coin_names = upbit.get_all_coin_names()

    for coin_name in coin_names:
        # hidden_size = 256
        # output_size = 2
        # model = LSTM(input_size, hidden_size, output_size, num_layers=3).to(device)
        model = CNN().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        valid_losses = []

        avg_train_losses = []
        avg_valid_losses = []

        train_accuracy_list = []
        valid_accuracy_list = []

        patience = 50

        early_stopping = EarlyStopping(coin_name=coin_name, patience=patience, verbose=True)

        upbit_data = Upbit_Data(coin_name, train_cols)
        x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, one_rate_train, train_size, \
        x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, one_rate_valid, valid_size = upbit_data.get_data(
            coin_name=coin_name,
            windows_size=WINDOW_SIZE,
            future_target_size=FUTURE_TARGET_SIZE,
            up_rate=UP_RATE,
            cnn=True
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_data_loader = get_data_loader(
                x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, batch_size=batch_size, suffle=True
            )

            correct = 0.0
            total = 0.0

            #### training
            for i, (x_train, x_train_normalized, y_train, y_train_normalized, y_up_train, num_batches) in enumerate(train_data_loader):
                optimizer.zero_grad()
                out = model.forward(x_train_normalized)
                loss = criterion(out, y_up_train)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                _, output_index = torch.max(out, 1)
                total += y_up_train.size(0)
                correct += (output_index == y_up_train).sum().float()

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            train_accuracy = 100 * correct / total
            train_accuracy_list.append(train_accuracy)

            # 배치정규화나 드롭아웃은 학습할때와 테스트 할때 다르게 동작하기 때문에 모델을 evaluation 모드로 바꿔서 테스트해야합니다.
            model.eval()
            correct = 0.0
            total = 0.0

            valid_data_loader = get_data_loader(
                x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, batch_size=batch_size, suffle=True
            )

            #### evaluation
            for x_valid, x_valid_normalized, y_valid, y_valid_normalized, y_up_valid, num_batches in valid_data_loader:
                out = model.forward(x_valid_normalized)
                _, output_index = torch.max(out, 1)
                loss = criterion(out, y_up_valid)
                valid_losses.append(loss.item())

                total += y_up_valid.size(0)
                correct += (output_index == y_up_valid).sum().float()

            valid_accuracy = 100 * correct / total
            valid_accuracy_list.append(valid_accuracy)

            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)

            if epoch % 10 == 0:
                print_msg = "{0} - Epoch[{1}/{2}] train_loss:{3:.6f}, train_accuracy:{4:.2f}, valid_loss:{5:.6f}, valid_accuracy:{6:.2f}".format(
                    coin_name,
                    epoch,
                    NUM_EPOCHS,
                    train_loss,
                    train_accuracy,
                    valid_loss,
                    valid_accuracy
                )
                print(print_msg)

            early_stopping(valid_loss, epoch, model, valid_size, one_rate_valid)

            if early_stopping.early_stop:
                print("Early stopping @ Epoch {0}: Last Save Epoch {1}".format(epoch, early_stopping.last_save_epoch))
                break

        if epoch == NUM_EPOCHS:
            print("Normal Stopping @ Epoch {0}: Last Save Epoch {1}".format(epoch, early_stopping.last_save_epoch))

        save_graph(
            coin_name,
            early_stopping.val_loss_min,
            early_stopping.last_save_epoch,
            valid_size, one_rate_valid,
            avg_train_losses, train_accuracy_list, avg_valid_losses, valid_accuracy_list
        )

        print()
