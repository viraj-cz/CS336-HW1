import torch
from einops import einsum, rearrange
import math

#Note: modern LMs do not use bias for their linear layers!
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        empty_tensor = torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device)
        std_dev = math.sqrt(2/(self.in_features+self.out_features))
        initialized_weights = torch.nn.init.trunc_normal_(
            tensor=empty_tensor,
            mean=0,
            std=std_dev,
            a= -3 * std_dev,
            b= 3 * std_dev
        )
        self.W = torch.nn.parameter.Parameter(
            initialized_weights
        )
    
    def forward(self, x: torch.Tensor):
        #note to self:
        #here we use: x, self.W, "... d_in, d_in d_out -> ... d_out"
        #and not: x, self.W, "batch d_in, d_in d_out -> batch d_out"
        #because in first case it is more robust and work both with and without a batch, i.e a single op and batched ops
        return einsum(
            x, self.W, "... d_in, d_out d_in -> ... d_out"
        )


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        empty_tensor = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device)
        initialized_embeddings = torch.nn.init.trunc_normal_(
            tensor=empty_tensor,
            mean=0,
            std=1,
            a= -3,
            b= 3
        )
        self.W = torch.nn.parameter.Parameter(
            initialized_embeddings
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #Note: the indexing in PyTorch is quite fancy and can handle batched inp indexing
        return self.W[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.gain = torch.nn.parameter.Parameter(
            torch.randn(self.d_model, device=self.device, dtype=self.dtype)
        )
    
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_squared = x ** 2
        x_squared_plus_eps = x_squared + self.eps
        unscaled_RMS = einsum(x_squared_plus_eps, "batch_size seq_length d_model -> batch_size seq_length")
        RMS = torch.sqrt(1/self.d_model * unscaled_RMS)
        RMS_scaled = rearrange(RMS, "batch_size seq_length -> batch_size seq_length 1") #I fucking love einops
        result = x/RMS_scaled * self.gain

        return result.to(in_dtype)


