import numpy as np
import torch
import optax
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from .linear_operator_base import LinearOperator, Lazy
from .linear_operators import ConcatLazy, I, lazify, densify, LazyJVP
import logging
import matplotlib.pyplot as plt
from functools import reduce
from oil.utils.utils import export

from plum import dispatch
import emlp.reps
#TODO: add rep,v = flatten({'Scalar':..., 'Vector':...,}), to_dict(rep,vector) returns {'Scalar':..., 'Vector':...,}
#TODO and simpler rep = flatten({Scalar:2,Vector:10,...}),
# Do we even want + operator to implement non canonical orderings?

__all__ = ["V","Vector", "Scalar"]

@export
class Rep(object):
    r""" The base Representation class. Representation objects formalize the vector space V
        on which the group acts, the group representation matrix ρ(g), and the Lie Algebra
        representation dρ(A) in a single object. Representations act as types for vectors coming
        from V. These types can be manipulated and transformed with the built in operators
        ⊕,⊗,dual, as well as incorporating custom representations. Rep objects should
        be immutable.

        At minimum, new representations need to implement ``rho``, ``__str__``."""
        
    is_permutation=False

    def rho(self,M):
        """ Group representation of the matrix M of shape (d,d)"""
        raise NotImplementedError
        
    def drho(self,A): 
        """ Lie Algebra representation of the matrix A of shape (d,d)"""
        In = torc.eye(A.shape[0])
        return LazyJVP(self.rho,In,A)

    def __call__(self,G):
        """ Instantiate (non concrete) representation with a given symmetry group"""
        raise NotImplementedError

    def __str__(self): raise NotImplementedError 
    #TODO: separate __repr__ and __str__?
    def __repr__(self): return str(self)
    
    
    def __eq__(self,other):
        if type(self)!=type(other): return False
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        d2 = tuple([(k,v) for k,v in other.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return d1==d2
    def __hash__(self):
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return hash((type(self),d1))

    def size(self): 
        """ Dimension dim(V) of the representation """
        if hasattr(self,'_size'):
            return self._size
        elif self.concrete and hasattr(self,"G"):
            self._size = self.rho(self.G.sample()).shape[-1]
            return self._size
        else: raise NotImplementedError

    def rank(self):
        """ Rank of basis treated as equivariant """
        if hasattr(self, '_rank'):
            return self._rank
        else:
            raise NotImplementedError

    def canonicalize(self): 
        """ An optional method to convert the representation into a canonical form
            in order to reuse equivalent solutions in the solver. Should return
            both the canonically ordered representation, along with a permutation
            which can be applied to vectors of the current representation to achieve
            that ordering. """
        return self, np.arange(self.size()) # return canonicalized rep
    
    def rho_dense(self,M):
        """ A convenience function which returns rho(M) as a dense matrix."""
        return densify(self.rho(M))
    def drho_dense(self,A):
        """ A convenience function which returns drho(A) as a dense matrix."""
        return densify(self.drho(A))
    
    def constraint_matrix(self):
        """ Constructs the equivariance constrant matrix (lazily) by concatenating
        the constraints (ρ(hᵢ)-I) for i=1,...M and dρ(Aₖ) for k=1,..,D from the generators
        of the symmetry group. """
        n = self.size()
        constraints = []
        constraints.extend([lazify(self.rho(h))-I(n) for h in self.G.discrete_generators])
        constraints.extend([lazify(self.drho(A)) for A in self.G.lie_algebra])
        return ConcatLazy(constraints) if constraints else lazify(torch.zeros((1,n)))

    # TODO: is there a way to cache the results here?
    def equivariant_basis(self):  
        """ Returns the basis corresponding to the smallest k=self.rank() singular values,
            and a null_space_loss: the sum of the squares of said singular values. """
        if self==Scalar: return torch.ones((1,1)), 0
        k = self.rank()
        C_lazy = self.constraint_matrix()
        C_dense = C_lazy.to_dense()
        U, S, Vh = torch.linalg.svd(C_dense)
        null_space_loss = torch.linalg.norm(S[-k:])**2
        Q = Vh[-k:].conj().T
        return Q, null_space_loss
    
    def equivariant_projector(self):
        """ Computes the (lazy) projection matrix P=QQᵀ that projects to the equivariant basis."""
        Q, null_space_loss = self.equivariant_basis()
        Q_lazy = lazify(Q)
        P = Q_lazy@Q_lazy.H
        return P, null_space_loss

    @property
    def concrete(self):
        return hasattr(self,"G") and self.G is not None
        # if hasattr(self,"_concrete"): return self._concrete
        # else:
        #     return hasattr(self,"G") and self.G is not None

    def __add__(self, other):
        """ Direct sum (⊕) of representations. """
        if isinstance(other,int):
            if other==0: return self
            else: return self+other*Scalar
        elif emlp.reps.product_sum_reps.both_concrete(self,other):
            return emlp.reps.product_sum_reps.SumRep(self,other)
        else:
            return emlp.reps.product_sum_reps.DeferredSumRep(self,other)

    def __radd__(self,other):
        if isinstance(other,int): 
            if other==0: return self
            else: return other*Scalar+self
        else: return NotImplemented
        
    def __mul__(self,other):
        """ Tensor sum (⊗) of representations. """
        return mul_reps(self,other)
            
    def __rmul__(self,other):
        return mul_reps(other,self)

    def __pow__(self,other):
        """ Iterated tensor product. """
        assert isinstance(other,int), f"Power only supported for integers, not {type(other)}"
        assert other>=0, f"Negative powers {other} not supported"
        return reduce(lambda a,b:a*b,other*[self],Scalar)
    def __rshift__(self,other):
        """ Linear maps from self -> other """
        return other*self.T
    def __lshift__(self,other):
        """ Linear maps from other -> self """
        return self*other.T
    def __lt__(self, other):
        """ less than defined to disambiguate ordering multiple different representations.
            Canonical ordering is determined first by Group, then by size, then by hash"""
        if other==Scalar: return False
        try: 
            if self.G<other.G: return True
            if self.G>other.G: return False
        except (AttributeError,TypeError): pass
        if self.size()<other.size(): return True
        if self.size()>other.size(): return False
        return hash(self) < hash(other) #For sorting purposes only
    def __mod__(self,other): # Wreath product
        """ Wreath product of representations (Not yet implemented)"""
        raise NotImplementedError
    @property
    def T(self):
        """ Dual representation V*, rho*, drho*."""
        if hasattr(self,"G") and (self.G is not None) and self.G.is_orthogonal: return self
        return Dual(self)


@dispatch
def mul_reps(ra,rb:int):
    if rb==1: return ra
    if rb==0: return 0
    if (not hasattr(ra,'concrete')) or ra.concrete:
        return emlp.reps.product_sum_reps.SumRep(*(rb*[ra]))
    else:
        return emlp.reps.product_sum_reps.DeferredSumRep(*(rb*[ra]))

@dispatch
def mul_reps(ra:int,rb):
    return mul_reps(rb,ra)

# Continued with non int cases in product_sum_reps.py

# A possible
class ScalarRep(Rep):
    def __init__(self,G=None):
        self.G=G
        self.is_permutation = True
    def __call__(self,G):
        self.G=G
        return self
    def size(self):
        return 1
    def rank(self):
        return 0
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        return "V⁰"
    @property
    def T(self):
        return self
    def rho(self,M):
        return torch.eye(1)
    def drho(self,M):
        return 0*torch.eye(1)
    def __hash__(self):
        return 0
    def __eq__(self,other):
        return isinstance(other,ScalarRep)
    def __mul__(self,other):
        if isinstance(other,int): return super().__mul__(other)
        return other
    def __rmul__(self,other):
        if isinstance(other,int): return super().__rmul__(other)
        return other
    @property
    def concrete(self):
        return True

class Base(Rep):
    """ Base representation V of a group."""
    def __init__(self,rank,G=None):
        self.G=G
        self._rank=rank
        if G is not None: self.is_permutation = G.is_permutation
    def __call__(self,G):
        return self.__class__(self.rank(),G)
    def rho(self,M):
        if hasattr(self,'G') and isinstance(M,dict): M=M[self.G]
        return M
    def drho(self,A):
        if hasattr(self,'G') and isinstance(A,dict): A=A[self.G]
        return A
    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        #return "V"# +(f"_{self.G}" if self.G is not None else "")
        return f"Vᵣ₌{chr(0x2080 + self.rank())}"
    
    def __hash__(self):
        return hash((type(self),self.G))
    def __eq__(self,other):
        return type(other)==type(self) and self.G==other.G
    def __lt__(self,other):
        if isinstance(other,Dual): return True
        return super().__lt__(other)
    # @property
    # def T(self):
    #     return Dual(self.G)

class Dual(Rep):
    def __init__(self,rep):
        self.rep = rep
        self.G=rep.G
        if hasattr(rep,"is_permutation"): self.is_permutation = rep.is_permutation
    def __call__(self,G):
        return self.rep(G).T
    def rho(self,M):
        rho = self.rep.rho(M)
        rhoinvT = rho.invT() if isinstance(rho,LinearOperator) else torch.linalg.inv(rho).T
        return rhoinvT
    def drho(self,A):
        return -self.rep.drho(A).T
    def __str__(self):
        return str(self.rep)+"*"
    def __repr__(self): return str(self)
    @property
    def T(self):
        return self.rep
    def __eq__(self,other):
        return type(other)==type(self) and self.rep==other.rep
    def __hash__(self):
        return hash((type(self),self.rep))
    def __lt__(self,other):
        if other==self.rep: return False
        return super().__lt__(other)
    def size(self):
        return self.rep.size()
    def rank(self):
        return self.rep.rank()

V=Vector= (lambda rank: Base(rank))  #: Alias V or Vector for an instance of the Base representation of a group

Scalar = ScalarRep()#: An instance of the Scalar representation, equivalent to V**0

#@export
#def T(p,q=0,G=None):
#    """ A convenience function for creating rank (p,q) tensors."""
#    return (V**p*V.T**q)(G)
@export
def T(rank,p,q=0,G=None):
    """ A convenience function for creating rank (p,q) tensors."""
    return (V(rank)**p*V(rank).T**q)(G)

class ConvergenceError(Exception): pass

#@partial(jit,static_argnums=(0,1))
@export
def bilinear_weights(out_rep,in_rep):
    #TODO: replace lazy_projection function with LazyDirectSum LinearOperator
    W_rep,W_perm = (in_rep>>out_rep).canonicalize()
    inv_perm = np.argsort(W_perm)
    mat_shape = out_rep.size(),in_rep.size()
    x_rep=in_rep
    W_multiplicities = W_rep.reps
    x_multiplicities = x_rep.reps
    x_multiplicities = {rep:n for rep,n in x_multiplicities.items() if rep!=Scalar}
    nelems = lambda nx,rep: min(nx,rep.size())
    active_dims = sum([W_multiplicities.get(rep,0)*nelems(n,rep) for rep,n in x_multiplicities.items()])
    reduced_indices_dict = {rep:ids[np.random.choice(len(ids),nelems(len(ids),rep))].reshape(-1)\
                                for rep,ids in x_rep.as_dict(np.arange(x_rep.size())).items()}
    # Apply the projections for each rank, concatenate, and permute back to orig rank order
    def lazy_projection(params,x): # (r,), (*c) #TODO: find out why backwards of this function is so slow
        bshape = x.shape[:-1]
        x = x.reshape(-1,x.shape[-1])
        bs = x.shape[0]
        i=0
        Ws = []
        for rep, W_mult in W_multiplicities.items():
            if rep not in x_multiplicities:
                Ws.append(torch.zeros((bs,W_mult*rep.size())))
                continue
            x_mult = x_multiplicities[rep]
            n = nelems(x_mult,rep)
            i_end = i+W_mult*n
            bids =  reduced_indices_dict[rep]
            bilinear_params = params[i:i_end].reshape(W_mult,n) # bs,nK-> (nK,bs)
            i = i_end  # (bs,W_mult,d^r) = (W_mult,n)@(n,d^r,bs)
            bilinear_elems = bilinear_params@x[...,bids].T.reshape(n,rep.size()*bs)
            bilinear_elems = bilinear_elems.reshape(W_mult*rep.size(),bs).T
            Ws.append(bilinear_elems)
        Ws = torch.cat(Ws,dim=-1) #concatenate over rep axis
        return Ws[...,inv_perm].reshape(*bshape,*mat_shape) # reorder to original rank ordering
    return active_dims,lazy_projection
        
# @jit
# def mul_part(bparams,x,bids):
#     b = prod(x.shape[:-1])
#     return (bparams@x[...,bids].T.reshape(bparams.shape[-1],-1)).reshape(-1,b).T

@export
def vis(repin,repout,cluster=True):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G<S(n) but not true for many continuous groups)."""
    rep = (repin>>repout)
    P,_ = rep.equivariant_projector() # compute the equivariant basis
    Q,_ = rep.equivariant_basis()
    v = np.random.randn(P.shape[1])  # sample random vector
    v = np.round(P@v,decimals=4)  # project onto equivariant subspace (and round)
    if cluster: # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1,1)).labels_
    plt.imshow(v.reshape(repout.size(),repin.size()))
    plt.axis('off')


def scale_adjusted_rel_error(t1,t2,g):
    error = torch.sqrt(torch.mean(torch.abs(t1-t2)**2))
    tscale = torch.sqrt(torch.mean(torch.abs(t1)**2)) + torch.sqrt(torch.mean(torch.abs(t2)**2))
    gscale = torch.sqrt(torch.mean(torch.abs(g-torch.eye(g.shape[-1]))**2))
    scale = torch.maximum(tscale,gscale)
    return error/torch.maximum(scale,1e-7)

@export
def equivariance_error(W,repin,repout,G):
    """ Computes the equivariance relative error rel_err(Wρ₁(g),ρ₂(g)W)
        of the matrix W (dim(repout),dim(repin)) [or basis Q: (dim(repout)xdim(repin), r)]
        according to the input and output representations and group G. """
    W = W.reshape(repout.size(),repin.size(),-1).transpose((2,0,1))[None]

    # Sample 5 group elements and verify the equivariance for each
    gs = G.samples(5)
    ring = vmap(repin.rho_dense)(gs)[:,None]
    routg = vmap(repout.rho_dense)(gs)[:,None]
    equiv_err = scale_adjusted_rel_error(W@ring,routg@W,gs)
    return equiv_err

import emlp.groups # Why is this necessary to avoid circular import?
