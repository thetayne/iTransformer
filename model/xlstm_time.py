import torch.nn.functional as F
from torch import nn
import torch
from einops import rearrange

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig

mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()

config = xLSTMBlockStackConfig(
    mlstm_block=mlstm_config,
    slstm_block=slstm_config,
    num_blocks=3,
    embedding_dim=256,
    add_post_blocks_norm=True,
    _block_map=1,
    context_length=336
)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.context_length = configs.seq_len
        self.embedding_dim = configs.embedding_dim
        self.output_size = configs.c_out
        self.kernel_size = configs.kernal_size

        self.dropout = nn.Dropout(configs.dropout)

        # Decomposition Kernel Size
        kernel_size = configs.moving_avg
        self.decomposition = series_decomp2(kernel_size)
        self.Linear_Seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.Linear_Trend = nn.Linear(configs.seq_len, configs.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1 / configs.seq_len) * torch.ones([configs.pred_len, configs.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1 / configs.seq_len) * torch.ones([configs.pred_len, configs.seq_len]))

        self.mm = nn.Linear(configs.pred_len + configs.mark_enc_in, configs.embedding_dim)
        self.mm2 = nn.Linear(configs.embedding_dim, configs.c_out)

        self.xlstm_stack = xLSTMBlockStack(config)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        #x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        x_combined = x_enc

        seasonal_init, trend_init = self.decomposition(x_combined)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        #x = torch.cat((x, x_enc.permute(0, 2, 1)), dim=1)  # Add time encodings
        x = self.mm(x)
        x = self.xlstm_stack(x)
        x = self.mm2(x)

        x = self.dropout(x)
        out = x.permute(0, 2, 1)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp2(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
