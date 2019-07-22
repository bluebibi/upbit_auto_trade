import torch
import torch.nn as nn
from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bias=True, dropout=0.4, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.GRU):
                params = m.named_parameters()
                for name, params in params:
                    if "weight" in name:
                        init.kaiming_normal_(params.data)
                    if "bias" in name:
                        params.data.fill_(0)
        print("param initialized!!!")

    def forward(self, x):
        hidden, cell = self.init_hidden(x)

        out, (hidden, cell) = self.rnn(x, (hidden, cell))
        out = self.fc(out[:, -1, :])
        #print(out.size())
        out = self.softmax(out)

        return out

    def init_hidden(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        return hidden, cell