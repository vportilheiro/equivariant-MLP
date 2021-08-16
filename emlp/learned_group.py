import objax
from emlp.groups import Group


class LearnedGroup(Group,objax.Module):
    def __init__(self,d,ncontinuous=3,ndiscrete=3):
        self._d = d
        self._discrete_generators = objax.variable.TrainVar(objax.random.normal((ndiscrete,d,d)))
        self._lie_algebra = objax.variable.TrainVar(objax.random.normal((ncontinuous,d,d)))
        self.is_permutation = False
        self.is_orthogonal = False
        Group.__init__(self)
    @property
    def discrete_generators(self):
        return self._discrete_generators.value
    @property
    def lie_algebra(self):
        return self._lie_algebra.value
    @property
    def d(self):
        return self._d
