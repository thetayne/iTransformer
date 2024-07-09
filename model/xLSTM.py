import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.context_length = configs.seq_len
        self.input_size = configs.enc_in + configs.mark_enc_in
        self.embedding_dim = configs.embedding_dim
        self.output_size = configs.c_out
        self.kernal_size = configs.kernal_size
        self.num_heads = configs.num_heads
        self.qkv_proj_blocksize = configs.qkv_proj_blocksize
        self.proj_factor = configs.proj_factor
        self.num_blocks = configs.num_blocks
        self.slstm_at = configs.slstm_at


        self.dropout = nn.Dropout(configs.dropout)  # Dropout layer
        self.embedding = nn.Linear(self.input_size, self.embedding_dim)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.kernal_size, qkv_proj_blocksize=self.qkv_proj_blocksize, num_heads=self.num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=self.num_heads,
                    conv1d_kernel_size=self.kernal_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=self.proj_factor, act_fn="gelu"),
            ),
            context_length=self.context_length,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
            slstm_at=[self.slstm_at],  # Place sLSTM block at position 1
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.linear = nn.Linear(self.embedding_dim, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        x = self.embedding(x_combined)
        x = self.xlstm_stack(x)
        x = self.dropout(x) 
        out = self.linear(x[:, -self.pred_len:, :])
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out
