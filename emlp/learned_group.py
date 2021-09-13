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

def equivariance_loss(G, repin, repout, W=None, b=None, ord=2, normalize=False):
    if (not hasattr(repin, "G")) or repin.G is None:
        repin = repin(G)
    if (not hasattr(repout, "G")) or repout.G is None:
        repout = repout(G)
    loss = 0
    for h in G.discrete_generators:
        H_out = repout.rho_dense(h)
        if W is not None:
            H_in = repin.rho_dense(h)
            norm = jnp.linalg.norm(H_out @ W - W @ H_in, ord=ord)
            if normalize:
                norm = norm / (jnp.linalg.norm(W, ord=ord) + jnp.linalg.norm(H_in - jnp.eye(H_in.shape[0]), ord=ord) + jnp.linalg.norm(H_out - jnp.eye(H_out.shape[0]), ord=ord))
            loss += norm
        #loss += jnp.linalg.norm(((repin >> repout).rho_dense(h) @ W.reshape(-1)).reshape(W.shape) - W, ord=ord)

        # NOTE: we have to be quite careful here, since taking the gradient of a 2-norm when the vector
        # is 0 gives a NaN (https://github.com/google/jax/issues/3058). For this reason we just take the
        # square norm given by the inned product.
        if b is not None:
            diff = H_out @ b - b
            norm_sq = diff @ diff
            if normalize:
                norm_sq = norm_sq / (b @ b + jnp.linalg.norm(H_out - jnp.eye(H_out.shape[0]), ord=ord)**2)
            loss += norm_sq

    for A in G.lie_algebra:
        A_out = repout.drho_dense(A)
        if W is not None:
            A_in = repin.drho_dense(A)
            norm = jnp.linalg.norm(A_out @ W - W @ A_in, ord=ord)
            if normalize:
                norm = norm / (jnp.linalg.norm(W, ord=ord) + jnp.linalg.norm(A_in, ord=ord) + jnp.linalg.norm(A_out, ord=ord))
            loss += norm

        # NOTE: bias equivariance depends on the vector norm ||drho(A) b||, NOT ||drho(A) b - b||
        if b is not None:
            diff = A_out @ b
            norm_sq = diff @ diff
            if normalize:
                norm_sq = norm_sq / (b @ b + jnp.linalg.norm(A_out, ord=ord)**2)
            loss += norm_sq
    return loss

def data_fhat_equivariance_loss(G, repin, repout, x, y, fhat, ord=2):
    if (not hasattr(repin, "G")) or repin.G is None:
        repin = repin(G)
    if (not hasattr(repout, "G")) or repout.G is None:
        repout = repout(G)
    loss = 0
    for h in G.discrete_generators:
        H_in = repin.rho_dense(h)
        H_out = repout.rho_dense(h)
        loss += jnp.linalg.norm(
                (H_out @ y[...,jnp.newaxis]).squeeze() - fhat((H_in @ x[...,jnp.newaxis]).squeeze()), 
                ord=ord, axis=-1).mean()

    for A in G.lie_algebra:
        A_in = repin.drho_dense(A)
        A_out = repout.drho_dense(A)
        #loss += (y @ A_out.T - fhat(x @ A_in.T)).mean()
        loss += jnp.linalg.norm((A_out @ y[...,jnp.newaxis]).squeeze() - fhat((A_in @ x[...,jnp.newaxis]).squeeze()), ord=ord, axis=-1).mean()

    return loss

def generator_loss(G, ord=2):
        loss = 0
        for h in G.discrete_generators:
            loss += jnp.linalg.norm(h, ord=ord)
            # Penalizes being close to the identity
            #loss -= jnp.linalg.norm(h - jnp.eye(h.shape[-1]), ord=ord)
        for A in G.lie_algebra:
            loss += jnp.linalg.norm(A, ord=ord)
        return loss

