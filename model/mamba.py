import torch
import torch.nn as nn
from mamba_ssm import Mamba

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in + configs.mark_enc_in
        self.d_model = configs.d_model
        self.output_size = configs.c_out
        self.dropout = nn.Dropout(configs.dropout)  # Dropout layer

        self.input_transform = nn.Linear(self.input_size, self.d_model)

        self.mamba = Mamba(
            d_model=self.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand
        )
        self.fc = nn.Linear(self.d_model, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        x = self.input_transform(x_combined)
        x = self.mamba(x)
        x = self.dropout(x) 
        out = self.fc(x[:, -self.pred_len:, :])
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out
