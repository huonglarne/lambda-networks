import torch
from lambda_networks import LambdaLayer

layer = LambdaLayer(
    dim = 32,       # channels going in
    dim_out = 32,   # channels out
    n = 64,         # size of the receptive window - max(height, width)
    dim_k = 16,     # key dimension
    heads = 4,      # number of heads, for multi-query
    dim_u = 1       # 'intra-depth' dimension
)

x = torch.randn(1, 32, 64, 64)
layer(x) # (1, 32, 64, 64)