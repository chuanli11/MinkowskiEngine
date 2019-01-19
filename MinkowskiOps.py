import torch
from torch.nn.modules import Module
from SparseTensor import SparseTensor


class MinkowskiLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        output = self.linear(input.F)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)


def cat(sparse_tensors):
    """
    Given a tuple of sparse tensors, concatenate them.

    Ex) cat((a, b, c))
    """
    for s in sparse_tensors:
        assert isinstance(s, SparseTensor)
    coords_man = sparse_tensors[0].C
    coords_key = sparse_tensors[0].getKey().getKey()
    for s in sparse_tensors:
        assert coords_man == s.C
        assert coords_key == s.getKey().getKey()
    tens = []
    for s in sparse_tensors:
        tens.append(s.F)
    return SparseTensor(
        torch.cat(tens, dim=1),
        coords_key=sparse_tensors[0].getKey(),
        coords_manager=coords_man)
