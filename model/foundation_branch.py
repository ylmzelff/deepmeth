from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.project_config import DNABERT_HIDDEN_SIZE

CONV_OUT_CHANNELS = 1536
CONV_KERNEL_SIZE = 3
AVG_POOL_KERNEL_SIZE = 2
LSTM_HIDDEN_SIZE = 128
MHA_NUM_HEADS = 4
FOUNDATION_OUTPUT_DIM = LSTM_HIDDEN_SIZE * 2  # bidirectional LSTM + matching MHA embed_dim


class DeepMethFoundationBranch(nn.Module):
    
    def __init__(self, dropout_prob: float = 0.2):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=DNABERT_HIDDEN_SIZE,
            out_channels=CONV_OUT_CHANNELS,
            kernel_size=CONV_KERNEL_SIZE,
            padding=1,
        )
        self.avg_pool = nn.AvgPool1d(kernel_size=AVG_POOL_KERNEL_SIZE, stride=AVG_POOL_KERNEL_SIZE)
        self.batch_norm = nn.BatchNorm1d(CONV_OUT_CHANNELS)
        self.dropout_conv = nn.Dropout(p=dropout_prob)

        self.bilstm = nn.LSTM(
            input_size=CONV_OUT_CHANNELS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_lstm = nn.Dropout(p=dropout_prob)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=FOUNDATION_OUTPUT_DIM,
            num_heads=MHA_NUM_HEADS,
            batch_first=True,
        )

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(f"Expected token_embeddings shape [B, T, 768], got {tuple(token_embeddings.shape)}")
        if token_embeddings.size(-1) != DNABERT_HIDDEN_SIZE:
            raise ValueError(
                f"Expected DNABERT-2 embedding dimension {DNABERT_HIDDEN_SIZE}, got {token_embeddings.size(-1)}"
            )
        if attention_mask.shape != token_embeddings.shape[:2]:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must match "
                f"token_embeddings' [B, T] = {tuple(token_embeddings.shape[:2])}"
            )

        mask = attention_mask.unsqueeze(1).to(token_embeddings.dtype)  # [B, 1, T]

        
        x = token_embeddings.transpose(1, 2) * mask

      
        x = F.relu(self.conv1d(x))
        x = x * mask

       
        x = self.avg_pool(x)
        pooled_mask = (self.avg_pool(mask) > 0).to(x.dtype)  # [B, 1, L]
        x = x * pooled_mask

       
        x = self.dropout_conv(self.batch_norm(x))
        x = x * pooled_mask

       
        x = x.transpose(1, 2)
        pooled_mask = pooled_mask.squeeze(1)  # [B, L]
        lengths = pooled_mask.sum(dim=1).clamp(min=1).to(torch.int64)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed)
       
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        x = self.dropout_lstm(x)

        
        key_padding_mask = ~pooled_mask.bool()  
        attention_output, _ = self.multihead_attention(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask, need_weights=False,
        )

       
        valid = pooled_mask.unsqueeze(-1).to(attention_output.dtype) 
        summed = (attention_output * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1e-9)
        return summed / counts
