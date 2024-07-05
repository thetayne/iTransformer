import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0,pred_len=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.pred_len = pred_len

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -self.pred_len, :])
        return out
    


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_len, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)

        # Concatenate time features if provided
        if x_mark_enc is not None:
            x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)

        # Pass through LSTM layers
        out, _ = self.lstm(x_enc, (h0, c0))

        # Take the last 'pred_len' time steps from the LSTM output
        out = self.fc(out[:, -self.pred_len:, :])
        return out
