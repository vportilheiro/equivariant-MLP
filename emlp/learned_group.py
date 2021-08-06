import objax
from emlp.groups import Group
from objax.module import Module

class LearnedGroup(Group,Module):
    def __init__(self,d,ncontinuous=3,ndiscrete=3):
        self._d = d
        self._discrete_generators = objax.variable.TrainVar(objax.random.normal((ndiscrete,d,d)))
        self._lie_algebra = objax.variable.TrainVar(objax.random.normal((ncontinuous,d,d)))
        super().__init__()
    @property
    def discrete_generators(self):
        return self._discrete_generators.value
    @property
    def lie_algebra(self):
        return self._lie_algebra.value
    @property
    def d(self):
        return self._d
