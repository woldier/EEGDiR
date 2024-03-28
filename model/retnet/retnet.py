import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from model.retnet.retention import MultiScaleRetention
from model import AbstractDenoiser


class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False, drop_out=.0):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Dropout(p=drop_out),  # 添加dropout
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X

            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is


class DiR(AbstractDenoiser):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False, seq_len=512, mini_seq=16, drop_out=0.3):
        super().__init__(loss_function=nn.MSELoss())
        self.mini_seq = mini_seq
        if seq_len % mini_seq != 0:
            raise AttributeError("seq_len({}) % mini_seq({}) != 0".format(seq_len, mini_seq))

        self.patchfiy = nn.Sequential(
            nn.Linear(seq_len, mini_seq * hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_out),  # 添加dropout
            nn.Linear(mini_seq * hidden_dim, mini_seq * hidden_dim),
            nn.GELU()
        )

        self.retnet = RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim, drop_out=drop_out)

        self.outNet = nn.Sequential(
            nn.Linear(hidden_dim, seq_len // mini_seq),
            nn.Flatten(start_dim=1)  # 展平
        )

    def _inner_forward(self, x):
        if len(x.shape) != 2:
            raise AttributeError("input tensor shape must be [B,L]")
        x = self.patchfiy(x)
        x = torch.unsqueeze(x, 1)

        chunk = torch.chunk(x, self.mini_seq, dim=2)
        x = torch.cat(chunk, dim=1)

        x = self.retnet(x)

        out = self.outNet(x)
        return out


if __name__ == "__main__":
    net = DiR(8, 512, 1024, 8)
    input = torch.randn(16, 512)
    out, loss = net(input, input)
    print(out.shape)
