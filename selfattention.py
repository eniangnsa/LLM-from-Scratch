import torch.nn as nn
import torch
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        # compute the similarities : (q * k^T)
        sims = torch.matmult(q, k.Transpose(dim0=self.row_dim, dim1=self.col_dim))

        # scale similarities by dividing by sqrt(k.col_dim)
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmult(attention_percents, v)

        return attention_scores

