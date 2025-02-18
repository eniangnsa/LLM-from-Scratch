import torch 
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=0):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings, mask=None):

        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        similarities = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_similarities = similarities / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_similarities = scaled_similarities.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_similarities, dim=self.col_dim)

        attention_scores  = torch.matmul(attention_percents, v)

        return attention_scores


