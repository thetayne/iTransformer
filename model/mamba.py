import torch
import torch.nn as nn
from mamba_ssm import Mamba  # Ensure this is the correct import

class Model(nn.Module):
    """
    Mamba Model for Time Series Forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in + configs.mark_enc_in  # Add the size of the time features
        self.d_model = configs.d_model
        self.output_size = configs.c_out

        self.input_transform = nn.Linear(self.input_size, self.d_model)  # Adjust the embedding layer

        self.mamba = Mamba(
            d_model=self.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand
        )
        self.fc = nn.Linear(self.d_model, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Concatenate x_enc and x_mark_enc along the last dimension
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)

        x = self.input_transform(x_combined)
        x = self.mamba(x)
        
        out = self.fc(x[:, -self.pred_len:, :])
        
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out
