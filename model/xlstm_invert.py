import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig
from layers.Embed import DataEmbedding_inverted  

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.context_length = configs.seq_len
        self.input_size = configs.enc_in + configs.mark_enc_in
        self.embedding_dim = configs.embedding_dim
        self.output_size = configs.c_out
        self.kernel_size = configs.kernal_size
        self.num_heads = configs.num_heads_xlstm
        self.qkv_proj_blocksize = configs.qkv_proj_blocksize
        self.proj_factor = configs.proj_factor
        self.num_blocks = configs.num_blocks
        self.slstm_at = configs.slstm_at
        self.use_norm = configs.use_norm

        self.dropout = nn.Dropout(configs.dropout)  # Dropout layer
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, self.embedding_dim)

        # xLSTM configuration
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.kernel_size, qkv_proj_blocksize=self.qkv_proj_blocksize, num_heads=self.num_heads
                )
            ),
            context_length=self.context_length,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_dim,
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.linear = nn.Linear(self.embedding_dim, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        print(x_enc.shape)
        print(x_mark_enc.shape)
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # x_enc has torch.Size([16, 96, 321]), i.e. [B, L, N]
        # x_mark_enc has torch.Size([16, 96, 4]), i.e. [B, L, N]

        
        # Embedding
        x_enc = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, N] -> [B, L, E]
        # After embedding: torch.Size([16, 325, 256])

        # Process with xLSTM stack
        x_enc = self.xlstm_stack(x_enc)
        x_enc = self.dropout(x_enc)

        # Linear projection
        dec_out = self.linear(x_enc[:, -self.pred_len:, :])  # [B, L, E] -> [B, pred_len, output_size]

        if self.use_norm:
            # De-normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]  # [B, L, D]
