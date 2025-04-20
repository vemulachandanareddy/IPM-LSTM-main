import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):

        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):

        # nn.MultiheadAttention expects input in shape [seq_len, batch_size, embed_dim]
        x_transposed = x.transpose(0, 1)
        attn_output, attn_weights = self.multihead_attn(x_transposed, x_transposed, x_transposed)
        # Transpose back to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights
