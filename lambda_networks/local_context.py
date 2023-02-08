import torch
from lambda_networks import LambdaLayer

layer = LambdaLayer(
    dim = 32,
    dim_out = 32,
    r = 23,         # the receptive field for relative positional encoding (23 x 23)
    dim_k = 16,
    heads = 4,
    dim_u = 4
)

x = torch.randn(1, 32, 64, 64)
layer(x) # (1, 32, 64, 64)