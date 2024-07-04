import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    LSTM Model for Time Series Forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in
        self.output_size = configs.c_out

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=configs.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # LSTM doesn't need the decoder input, so we only use x_enc
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)

        out, _ = self.lstm(x_enc, (h0, c0))
        out = self.fc(out[:, -self.pred_len:, :])
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
