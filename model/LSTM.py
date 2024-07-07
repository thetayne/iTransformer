import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in + configs.mark_enc_in
        self.output_size = configs.c_out
        self.dropout = nn.Dropout(configs.dropout)  # Dropout layer

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=configs.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        h0 = torch.zeros(self.num_layers, x_combined.size(0), self.hidden_size).to(x_combined.device)
        c0 = torch.zeros(self.num_layers, x_combined.size(0), self.hidden_size).to(x_combined.device)
        out, _ = self.lstm(x_combined, (h0, c0))
        out = self.fc(out[:, -self.pred_len:, :])
        out = self.dropout(out)  # Apply dropout after the linear layer
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out


    




    # def __init__(self, configs):
    #     super(Model, self).__init__()
    #     self.hidden_size = configs.hidden_size
    #     self.num_layers = configs.num_layers
    #     self.pred_len = configs.pred_len
    #     self.input_size = configs.enc_in
    #     self.output_size = configs.c_out

    #     self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=configs.dropout)
    #     self.fc = nn.Linear(self.hidden_size, self.output_size)

    # def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # LSTM doesn't need the decoder input, so we only use x_enc
    #     h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)
    #     c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device)

    #     out, _ = self.lstm(x_enc, (h0, c0))
    #     out = self.fc(out[:, -self.pred_len:, :])
    #     return out

    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    #     dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    #     return dec_out
