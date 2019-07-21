import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = self.init_hidden()

        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.view(batch_size, -1))
        return out, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden


if __name__ == "__main__":
    input_size = 11
    hidden_size = 256
    output_size = 2

    model = RNN(input_size, hidden_size, output_size, num_layers=3)