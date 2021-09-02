import jax.numpy as jnp
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

def equivariance_loss(G, repin, repout, W=None, b=None, ord=2):
    repin = repin(G)
    repout = repout(G)
    loss = 0
    for h in G.discrete_generators:
        if W is not None:
            loss += jnp.linalg.norm(repout.rho_dense(h) @ W - W @ repin.rho_dense(h), ord=ord)
        #loss += jnp.linalg.norm(((repin >> repout).rho_dense(h) @ W.reshape(-1)).reshape(W.shape) - W, ord=ord)

        # NOTE: we have to be quite careful here, since taking the gradient of a 2-norm when the vector
        # is 0 gives a NaN (https://github.com/google/jax/issues/3058). For this reason we just take the
        # square norm given by the inned product.
        if b is not None:
            diff = (repout.rho_dense(h) @ b - b)
            loss += diff @ diff

    for A in G.lie_algebra:
        if W is not None:
            loss += jnp.linalg.norm(repout.drho_dense(A) @ W - W @ repin.drho_dense(A), ord=ord)

        # NOTE: bias equivariance depends on the vector norm ||drho(A) b||, NOT ||drho(A) b - b||
        if b is not None:
            diff = (repout.drho_dense(A) @ b)
            loss += diff @ diff
    return loss


