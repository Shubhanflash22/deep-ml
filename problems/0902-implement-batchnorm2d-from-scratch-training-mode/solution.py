import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    C = gamma.shape[0]
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var  = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

    x_norm = (x - mean) / torch.sqrt(var + eps)

    y = gamma.view(1, C, 1, 1) * x_norm + beta.view(1, C, 1, 1)
    return y