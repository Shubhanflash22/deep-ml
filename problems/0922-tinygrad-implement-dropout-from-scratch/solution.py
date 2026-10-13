from tinygrad import Tensor

def dropout(x, p, training):
    if training == False or p <= 0:
        return x
    else:
        noise = Tensor.rand(*x.shape)
        mask = (noise >= p).cast(x.dtype)
    return x * mask / (1 - p)