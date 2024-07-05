import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

class Model(nn.Module):
    """
    xLSTM Model for Time Series Forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in + configs.mark_enc_in  # Add the size of the time features
        self.embedding_dim = configs.embedding_dim
        self.output_size = configs.c_out

        self.embedding = nn.Linear(self.input_size, self.embedding_dim)  # Adjust the embedding layer

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=configs.context_length,
            num_blocks=7,
            embedding_dim=self.embedding_dim,
            slstm_at=[1],  # Place sLSTM block at position 1
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.linear = nn.Linear(self.embedding_dim, self.output_size)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Concatenate x_enc and x_mark_enc along the last dimension
        x_combined = torch.cat((x_enc, x_mark_enc), dim=-1)
        
        # Debugging statements
        #print(f"x_enc shape: {x_enc.shape}")
        #print(f"x_mark_enc shape: {x_mark_enc.shape}")
        #print(f"x_combined shape: {x_combined.shape}")
        
        x = self.embedding(x_combined)
        print(f"Embedding output shape: {x.shape}")
        
        x = self.xlstm_stack(x)
        #print(f"xLSTM stack output shape: {x.shape}")
        
        out = self.linear(x[:, -self.pred_len:, :])
        #print(f"Linear layer output shape: {out.shape}")
        
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out

