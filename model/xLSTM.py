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
        self.num_heads = configs.num_heads_xlstm
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
            context_length=self.context_length,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
            #slstm_at=[self.slstm_at],  # Place sLSTM block at position 1
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.linear = nn.Linear(self.embedding_dim, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        print(x_enc.shape)
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        print(x_combined.shape)
        x = self.embedding(x_combined)
        print(x.shape)
        x = self.xlstm_stack(x)
        print(x.shape)
        x = self.dropout(x) 
        out = self.linear(x[:, -self.pred_len:, :])
        print(out.shape)
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out
