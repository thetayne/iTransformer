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
        print(f"x_combined shape: {x_combined.shape}")  # Debugging statement

        x = self.input_transform(x_combined)
        print(f"Transformed input shape: {x.shape}")  # Debugging statement

        x = self.mamba(x)
        print(f"Mamba output shape: {x.shape}")  # Debugging statement

        out = self.fc(x[:, -self.pred_len:, :])
        print(f"Final output shape: {out.shape}")  # Debugging statement

        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
