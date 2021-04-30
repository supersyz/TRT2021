import torch
import itertools
import operator

# https://github.com/onnx/onnx-tensorrt/issues/506


def gather(input, dim, index):
    indices = [torch.arange(size, device=index.device) for size in index.shape]
    indices = list(torch.meshgrid(*indices))
    indices[dim] = index
    sizes = list(
        reversed(list(itertools.accumulate(reversed(input.shape), operator.mul))))
    index = sum((index * size for index,
                 size in zip(indices, sizes[1:] + [1])))
    output = input.flatten()[index]
    return output
