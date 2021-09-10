import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp.learned_group import equivariance_loss
from emlp.reps import T,Rep,Scalar
from emlp.reps import bilinear_weights
from emlp.reps.product_sum_reps import SumRep
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from scipy.special import binom
from jax import jit,vmap
from functools import lru_cache as cache
from collections import defaultdict

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return nn.Sequential(args)

@export
class Network(Module):
    def __init__(self, G, layers):
        self.G = G
        self.network = Sequential(*layers)
    def __call__(self,x):
        return self.network(x)

@export
class Linear(Module):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, repin, repout, use_bias=True):
        self.repin, self.repout = repin, repout
        nin,nout = repin.size(),repout.size()
        self.use_bias = use_bias
        #super().__init__(nin,nout)
        self.rep_W = rep_W = repout*repin.T
        self.rep_bias = rep_bias = repout

        self.W_pre_proj = TrainVar(orthogonal((nout, nin)))
        if use_bias:
            self.b_pre_proj = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        else:
            self.b_pre_proj = jnp.zeros((nout,))

        self.Pw = rep_W.equivariant_projector()
        self.Pb = rep_bias.equivariant_projector()

        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")

    @property
    def W(self):
        #vec_W_pre_proj = self.W_pre_proj.reshape(-1)
        #vec_W = self.Pw @ vec_W_pre_proj
        #W_shape = self.W_pre_proj.shape
        #W = vec_W.reshape(W_shape)
        #return W
        return (self.Pw @ self.W_pre_proj.reshape(-1)).reshape(*self.W_pre_proj.shape)

    @property
    def b(self):
        return self.Pb @ self.b_pre_proj#.value

    def __call__(self, x): # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        out = x @ self.W.T + self.b
        logging.debug(f"linear out shape:{out.shape}")
        return out

    def equivariance_loss(self,G,ord=2):
        return equivariance_loss(G, self.repin, self.repout, self.W, self.b, ord)

@export
class ProjectionRecomputingLinear(Linear):
    """ A linear layer which recomputes the projectors Pw and Pb on each access to W and b respectively.
        Note that Pw and Pb are computed using the "approximately_equivariant" method. This method
        takes a "singular value weight function" (sv_weight_func) and then projects onto the right
        singular vectors of the constraint matrix, scaled by this weight function. (See for example
        how the RankK subclass projects using the k smallest singular values.) This layer stores
        the singular values and weights as values (tuples) sv_w_dict, whose keys are the representations
        for which these values are calculated --- which will be either rep_W or rep_bias (or sub-reps 
        thereof). The projection_loss() measures how far away the projections are from actually
        projecting onto the null-spaces of the contraints. (High weights should only be given
        to small singular values.)
        Note: the sv_offset parameter is used to specify a small value by which to offset the contraint
        matrix in order to keep it well conditioned. Otherwise, backprop through SVD gives NaNs. NaNs
        may still appear, and currently the best guess is because this doesn't totally fix the bad
        conditioning. (See the usage of the offset parameter in Rep.approximate_equivariant_basis().)

        The layer also allows for defining a "singular value loss", which imposes an inductive bias
        on the singular values themselves. For examples, having a loss function lambda S: jnp.sum(S) is
        a particular way of penalizing large singular values. Note however that it may be desireable to
        penalize singular values for different representations differently. For example, if we consider
        the group SO(2), a vector is invariant iff is it the zero vector, while an equivariant linear
        map can be any rotation. Thus if repin=repout=V, rep_bias should have all large singular values,
        while rep_W should have some zero singular values. This illustrates that penalizing singular
        values equally across all representations may be suboptimal (in fact in this case it leads
        to the learned symmetry being trival).

        We thus have a sv_loss_dict parameter, which we use to initialize a defaultdict, from representation
        to loss function. If a representation isn't in the passed parameter, the sv_loss_func is used.
        """
    #TODO(?): implement a similar system for different sv_weight_funcs across different representations
    def __init__(self, repin, repout, sv_weight_func=None, sv_offset=1e-4,
                 sv_loss_func=None, sv_loss_dict=None, **kwargs):
        # Parameters dealing with projection
        self.sv_weight_func = sv_weight_func
        self.sv_offset = sv_offset

        self.sv_w_dict = {}

        # Parameters for expressing inductive biases about singular values
        self.sv_loss_func = sv_loss_func
        self.sv_loss_dict = defaultdict(lambda: lambda S: self.sv_loss_func(S)) # Note: the extra lambda is not a mistake
        if sv_loss_dict is not None:
            self.sv_loss_dict.update(sv_loss_dict)
        super().__init__(repin,repout,**kwargs)

    @property
    def W(self):
        self.Pw, sv_w_W = self.rep_W.approximately_equivariant_projector(self.sv_weight_func, return_sv=True, offset=self.sv_offset)
        self.sv_w_dict.update(sv_w_W)
        return super().W

    @property
    def b(self):
        # We will not calculate any projectors or losses for the bias if use_bias=False
        if self.use_bias:
            self.Pb, sv_w_b = self.rep_bias.approximately_equivariant_projector(self.sv_weight_func, return_sv=True, offset=self.sv_offset)
            self.sv_w_dict.update(sv_w_b)
        return super().b

    def projection_loss(self):
        return sum((S*proj_weight) @ (S*proj_weight) for (S,proj_weight) in self.sv_w_dict.values())

    def sv_loss(self):
        return sum(self.sv_loss_dict[rep](S) for rep,(S,_) in self.sv_w_dict.items())


@export
class NoOptProjectionRecomputingLinear(Linear):
    """ The idea for this layer is to be like ProjectionRecomputingLinear, but not using any of the
        SVD decomposition across sum/product representations, instead solving the whole constraint
        across all the representations directly.
        NOTE: currently does not work, since SumReps do not have a self.G """
    def __init__(self, repin, repout, sv_weight_func=None, **kwargs):
        self.sv_weight_func = sv_weight_func
        self.sv_w_dict = {}
        super().__init__(repin,repout, **kwargs)

    @property
    def W(self):
        self.Pw, sv_w_W = Rep.approximately_equivariant_projector(self.rep_W, self.sv_weight_func, return_sv=True)
        self.sv_w_dict.update(sv_w_W)
        return super().W

    @property
    def b(self):
        self.Pb, sv_w_b = Rep.approximately_equivariant_projector(self.rep_bias, elf.sv_weight_func, return_sv=True)
        self.sv_w_dict.update(sv_w_b)
        return super().b

@export
def RankKLinear(k, **kwargs_outer):
    return lambda repin, repout, **kwargs: RankK(repin, repout, k, **{**kwargs, **kwargs_outer})
class RankK(ProjectionRecomputingLinear):
    def __init__(self, repin, repout, k, **kwargs):
        super().__init__(repin, repout, \
                lambda S: jax.lax.dynamic_update_slice(jnp.zeros_like(S), jnp.ones(k), [-k]),
                **kwargs)

@export
def SoftSVDLinear(cutoff, **kwargs_outer):
    return lambda repin, repout, **kwargs: SoftSVD(repin, repout, cutoff, **{**kwargs, **kwargs_outer})
class SoftSVD(ProjectionRecomputingLinear):
    def __init__(self, repin, repout, cutoff, **kwargs):
        super().__init__(repin, repout, \
                lambda S: jnp.exp(-0.5 * S**2 / (cutoff/3)**2),
                **kwargs)

@export
class ApproximatingLinear(objax.Module):
    """ A vanilla linear layer from repin to repout which knows how to calculate
        an "approximate equivariance loss". """
    def __init__(self, repin, repout, use_bias=True):
        self.repin, self.repout = repin, repout
        self.use_bias = use_bias
        self.rep_W, self.rep_bias = (repin >> repout), repout
        nin,nout = repin.size(),repout.size()
        #super().__init__(nin,nout)
        self._W = TrainVar(orthogonal((nout, nin)))
        if use_bias: self._b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))

    def __call__(self, x):
        return x @ self.W.T + self.b

    @property
    def W(self):
        return self._W.value

    @property
    def b(self):
        if self.use_bias:
            return self._b.value
        return jnp.zeros((self.repout.size(),))
    
    def equivariance_loss(self,G,ord=2):
        return equivariance_loss(G, self.repin, self.repout, self.W, self.b, ord)

    
@export
class BiLinear(Module):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout,repin)
        self.weight_proj = jit(weight_proj)
        self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
        logging.info(f"BiW components: dim:{Wdim}")

    def __call__(self, x,training=True):
        # compatible with non sumreps? need to check
        W = self.weight_proj(self.w.value,x)
        out= .1*(W@x[...,None])[...,0]
        return out

@export
def gated(sumrep): #TODO: generalize to mixed tensors?
    """ Returns the rep with an additional scalar 'gate' for each of the nonscalars and non regular
        reps in the input. To be used as the output for linear (and or bilinear) layers directly
        before a :func:`GatedNonlinearity` to produce its scalar gates. """
    return sumrep+sum([Scalar(rep.G) for rep in sumrep if rep!=Scalar and not rep.is_permutation])

@export
class GatedNonlinearity(Module): #TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        #activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        # XXX TODO NOTE: undo this change
        activations = jnp.exp(gate_scalars) * values[..., :self.rep.size()]
        return activations

@export
class EMLPBlock(Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    def __init__(self,rep_in,rep_out,LinearLayer=Linear):
        super().__init__()
        self.linear = LinearLayer(rep_in,gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)
    def __call__(self,x):
        lin = self.linear(x)
        preact =self.bilinear(lin)+lin
        return self.nonlinearity(preact)

def uniform_rep_general(ch,*rep_types):
    """ adds all combinations of (powers of) rep_types up to
        a total of ch channels."""
    #TODO: write this function
    raise NotImplementedError

@export
def uniform_rep(ch,group):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. Useful for hands off layer construction.
        
        Args:
            ch (int): total number of channels
            group (Group): symmetry group

        Returns:
            SumRep: The direct sum representation with dim(V)=ch
        """
    d = group.d
    Ns = np.zeros((lambertW(ch,d)+1,),int) # number of tensors of each rank
    while ch>0:
        max_rank = lambertW(ch,d) # compute the max rank tensor that can fit up to
        Ns[:max_rank+1] += np.array([d**(max_rank-r) for r in range(max_rank+1)],dtype=int)
        ch -= (max_rank+1)*d**max_rank # compute leftover channels
    sum_rep = sum([binomial_allocation(nr,r,group) for r,nr in enumerate(Ns)])
    sum_rep,perm = sum_rep.canonicalize()
    return sum_rep

def lambertW(ch,d):
    """ Returns solution to x*d^x = ch rounded down."""
    max_rank=0
    while (max_rank+1)*d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank

def binomial_allocation(N,rank,G):
    """ Allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r to match the binomial distribution.
        For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    n_binoms = N//(2**rank)
    n_leftover = N%(2**rank)
    even_split = sum([n_binoms*int(binom(rank,k))*T(k,rank-k,G) for k in range(rank+1)])
    ps = np.random.binomial(rank,.5,n_leftover)
    ragged = sum([T(int(p),rank-int(p),G) for p in ps])
    out = even_split+ragged
    return out

def uniform_allocation(N,rank):
    """ Uniformly allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r. For orthogonal representations there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    even_split = sum((N//(rank+1))*T(k,rank-k) for k in range(rank+1))
    ragged = sum(random.sample([T(k,rank-k) for k in range(rank+1)],N%(rank+1)))
    return even_split+ragged

@export
class EMLP(Module,metaclass=Named):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3,LinearLayer=Linear):#@
        super().__init__()
        logging.info("Initing EMLP (objax)")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        
        self.G=group
        self.num_generators = len(group.discrete_generators) + len(group.lie_algebra)
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout,LinearLayer) for rin,rout in zip(reps,reps[1:])],
            LinearLayer(reps[-1],self.rep_out)
        )
    def __call__(self,x,training=True):
        return self.network(x)

    def equivariance_loss(self, G=None, ord=2):
        L = len(self.network)
        if G is None:
            G = self.G
            num_generators = self.num_generators
        else:
            num_generators = len(G.discrete_generators) + len(G.lie_algebra)
        eq_loss = 0
        for l, layer in enumerate(self.network):
            if l < L - 1: linear = layer.linear
            else: linear = layer
            eq_loss += linear.equivariance_loss(G, ord=ord) / (L * num_generators) 
        return eq_loss

    def parameter_norm_sum(self, ord=2):
        L = len(self.network)
        norm_sum = 0
        for l, layer in enumerate(self.network):
            if l < L - 1: linear = layer.linear
            else: linear = layer
            norm_sum += jnp.linalg.norm(linear.W, ord=ord) / L
            norm_sum += jnp.linalg.norm(linear.b, ord=ord) / L
        return norm_sum


def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout):
    return Sequential(nn.Linear(cin,cout),swish)#,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,x,training=True):
        y = self.net(x)
        return y

@export
class Standardize(Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 

        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats
        
        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats
    def __call__(self,x,training):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin,training=training)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = sout*self.model((x-muin)/sin,training=training)+muout
            return y



# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPode(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,z,t):
        return self.net(z)

@export
class EMLPode(EMLP):
    """ Neural ODE Equivariant MLP. Same args as EMLP."""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        #super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #print(middle_layers[0].reps[0].G)
        #print(self.rep_in.G)
        reps = [self.rep_in]+middle_layers
        logging.info(f"Reps: {reps}")
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def __call__(self,z,t):
        return self.network(z)

# Networks for hamiltonian dynamics (need to sum for batched Hamiltonian grads)
@export
class MLPH(Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = Sequential(
            *[Sequential(nn.Linear(cin,cout),swish) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def H(self,x):#,training=True):
        y = self.net(x).sum()
        return y
    def __call__(self,x):
        return self.H(x)

@export
class EMLPH(EMLP):
    """ Equivariant EMLP modeling a Hamiltonian for HNN. Same args as EMLP"""
    #__doc__ += EMLP.__doc__.split('.')[1]
    def H(self,x):#,training=True):
        y = self.network(x)
        return y.sum()
    def __call__(self,x):
        return self.H(x)

@export
@cache(maxsize=None)
def gate_indices(sumrep): #TODO: add support for mixed_tensors
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    assert isinstance(sumrep,SumRep), f"unexpected type for gate indices {type(sumrep)}"
    channels = sumrep.size()
    perm = sumrep.perm
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rep in sumrep:
        if rep!=Scalar and not rep.is_permutation:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars+=1
        i+=rep.size()
    return indices
