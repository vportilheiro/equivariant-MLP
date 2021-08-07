import torch
import torch.nn as nn
from emlp.groups import Group


class LearnedGroup(Group,nn.Module):
    def __init__(self,d,ncontinuous=3,ndiscrete=3):
        nn.Module.__init__(self)
        self._d = d
        self._discrete_generators = nn.Parameter(torch.normal(0,1,size=(ndiscrete,d,d)))
        self._lie_algebra = nn.Parameter(torch.normal(0,1,size=(ncontinuous,d,d)))
        self.is_permutation = False
        self.is_orthogonal = False
        Group.__init__(self)
    @property
    def discrete_generators(self):
        return self._discrete_generators
    @property
    def lie_algebra(self):
        return self._lie_algebra
    @property
    def d(self):
        return self._d
