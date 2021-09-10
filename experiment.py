#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

# Logging must be set up before any JAX imports
import logging
logging.basicConfig(filename='experiment.log', encoding='utf-8', level=logging.INFO)

import emlp
from emlp.learned_group import LearnedGroup, equivariance_loss, generator_loss
from emlp.groups import S, Z, SO, O, Trivial
from emlp.reps import V, equivariance_error
import emlp.nn as nn

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import objax
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm, trange

# TODO: why does svd lead to multiple copies of/same generators?
# NOTE: check "normalize" option in equivariance_loss

##### Debuggin options #####

from jax.config import config
config.update("jax_debug_nans", True) # For tracing where NaNs come from
config.update('jax_disable_jit', False) # For examining values inside functions

##### Training hyperparameters #####

reg_eq = 0.3        # regularization parameter: how much to weight equivariance loss 
reg_gen = 0.0 #1e-8 # regularization parameter: how much to weight generator loss
reg_proj = 0.00     # regularization parameter: how much to weight projection loss
reg_sv = 0.00       # how much to weight loss from singular values
lr = 1e-4
epochs = 4*40000

# We define the dataset size by the number of batches and batch size.
# We have different kinds of batches: "all" for those used to update all parameters,
# "weights" for those updating only model weights, "gen" for updating only Ghat's generators.
batch_size = 8
dataset_size = 1 * batch_size 
batch_types = ["all", "weights", "gen"] # each epoch trains in this order
num_batches = {"all": 10000, "weights": 0, "gen": 0}
num_batches = {batch_type: min(num, dataset_size//batch_size) for batch_type, num in num_batches.items()}
num_val_batches = 1     # Used for validation every epoch. Currently, validation data is resampled each epoch.

resample_data = 40000          # How many epochs before a new dataset is sampled
resample_W = 40000             # How many epochs before a new W is sampled
reset_model_on_resample = True  # If true, the model weights are re-initialized on each resample of W

# Order of the matrix norm to use in losses
ord=2

# Model parameters
num_layers = 1
channels = V**0 + V

##### Task setup #####

n = 2
G = SO(n)
ncontinuous = len(G.lie_algebra)
ndiscrete = len(G.discrete_generators)
repin = V 
repout = V

def get_equivariant_W():
    Proj = (repin >> repout)(G).equivariant_projector()
    W = np.random.normal(size=(repout(G).size(),repin(G).size()))
    W = (Proj @ W.reshape(-1)).reshape(W.shape)
    return W

def equivariant_space_rank():
    return (repin >> repout)(G).equivariant_basis().shape[1]

def map_from_matrix(W):
    return lambda x: (W @ x[..., jnp.newaxis]).squeeze()

def main():

    ##### Initialize task ##### 

    W = get_equivariant_W()
    f = map_from_matrix(W)

    ##### Model and optimizer definition #####

    Ghat = LearnedGroup(n,ncontinuous,ndiscrete)
    #Ghat = G
    #Ghat = Trivial(n)
    ngenerators = ncontinuous + ndiscrete

    #model = nn.EMLP(repin, repout, \
    #        #LinearLayer=nn.SoftSVDLinear(1, sv_loss_func=lambda S: jnp.sum(jnp.tanh(S/5))), \
    #        LinearLayer=nn.ApproximatingLinear,
    #        group=Ghat, num_layers=num_layers, ch=channels)
    model = nn.Network(Ghat, [nn.ApproximatingLinear(repin(Ghat), repout(Ghat), use_bias=True)])
    #model = nn.Network(Ghat, [
    #    nn.SoftSVDLinear(1, use_bias=True,
    #                    sv_offset=1e-4,
    #                    sv_loss_func=lambda S: jnp.sum(jnp.tanh(S/5)),
    #                    sv_loss_dict={V(Ghat) :(lambda S: 0), V: (lambda S:0)},
    #                    )(repin(Ghat), repout(Ghat))
    #    ])
    #model = nn.Network(Ghat, [nn.RankKLinear(2, sv_loss_func=lambda S:0)(repin(Ghat), repout(Ghat), use_bias=False, sv_offset=1e-4)])

    all_vars = model.vars()
    gen_vars = objax.VarCollection((k,v) for k,v in all_vars.items() if "LearnedGroup" in k)
    weight_vars = objax.VarCollection((k,v) for k,v in all_vars.items() if "LearnedGroup" not in k)


    opt_all = objax.optimizer.Adam(all_vars)
    opt_weights = objax.optimizer.Adam(weight_vars)
    opt_gen = objax.optimizer.Adam(gen_vars)

    ##### objax functions #####

    @objax.Jit
    @objax.Function.with_vars(gen_vars)
    def Ghat_equivariance(W):
        return equivariance_loss(Ghat, repin, repout, W, ord=ord)

    @objax.Jit
    @objax.Function.with_vars(gen_vars)
    def Ghat_non_triviality():
        result = 0
        for h in Ghat.discrete_generators:
            result += jnp.linalg.norm(h - jnp.eye(h.shape[0]), ord=ord)
        for A in Ghat.lie_algebra:
            result += jnp.linalg.norm(A, ord=ord)
        return {"/non-triviality/Ghat": result / ngenerators}

    @objax.Jit
    @objax.Function.with_vars(all_vars)
    def loss(x, y):
        yhat = model(x)
        L = len(model.network)
        losses = defaultdict(float)
        for l, layer in enumerate(model.network):
            if l < L - 1: linear = layer.linear
            else: linear = layer
            # Ghat equivariance
            losses[f"layer {l}/equivariance/layer-Ghat"] = Ghat_eq = \
                    linear.equivariance_loss(Ghat, ord=ord) / ngenerators
            losses["/equivariance/model-Ghat"] += Ghat_eq / L

            # G equivariance
            # NOTE: do not use these losses for learning! They are for validation purposes only, as they
            # checks whether the model is equivariant under the unknown symmetry group G
            losses[f"layer {l}/equivariance/layer-G"] = G_eq = \
                    linear.equivariance_loss(G, ord=ord) / ngenerators
            losses["/equivariance/model-G"] += G_eq / L

            # weight norm
            losses[f"layer {l}/norm/What"] = W_norm = jnp.linalg.norm(linear.W, ord=ord)
            losses[f"/norm/What"] += W_norm / L

            if reg_proj > 0:
                losses["/svd/proj"] += linear.projection_loss() / L

            if reg_sv > 0:
                losses["/svd/sv"] += linear.sv_loss() / L

        losses["/prediction/train"] = ((yhat-y)**2).mean()
        losses["/norm/generators"] = generator_loss(Ghat, ord) / ngenerators
        p = losses["/svd/proj"] if reg_proj > 0 else 0
        s = losses["/svd/sv"] if reg_sv > 0 else 0
        m, e, g = losses["/prediction/train"], losses["/equivariance/model-Ghat"], \
                     losses["/norm/generators"]
        return ((1-reg_eq-reg_gen-reg_proj-reg_sv)*m+ reg_eq*e + reg_gen*g + reg_proj*p + reg_sv*s), losses

    @objax.Jit
    @objax.Function.with_vars(all_vars)
    def loss_with_W_equiv(x,y):
        training_loss, losses = loss(x,y)
        W_Ghat = Ghat_equivariance(W)
        losses["/equivariance/W-Ghat"] = W_Ghat
        return training_loss, losses

    grad_and_val_all = objax.GradValues(loss_with_W_equiv, all_vars)
    grad_and_val_weights = objax.GradValues(loss, weight_vars)
    grad_and_val_gen = objax.GradValues(loss_with_W_equiv, gen_vars)

    @objax.Jit
    @objax.Function.with_vars(all_vars+opt_all.vars())
    def train_op_all(x, y, lr):
        g, v = grad_and_val_all(x, y)
        opt_all(lr=lr, grads=g)
        return v

    @objax.Jit
    @objax.Function.with_vars(all_vars+opt_weights.vars())
    def train_op_weights(x, y, lr):
        g, v = grad_and_val_weights(x, y)
        opt_weights(lr=lr, grads=g)
        return v

    @objax.Jit
    @objax.Function.with_vars(all_vars+opt_gen.vars())
    def train_op_gen(x, y, lr):
        g, v = grad_and_val_gen(x, y)
        opt_gen(lr=lr, grads=g)
        return v

    train_ops = {"all": train_op_all, "weights": train_op_weights, "gen": train_op_gen}

    ##### print info #####

    print(f"model vars:\n{model.vars()}\n")
    print(f"True G: {G}")
    print(f"repin: {repin}, repout: {repout}")
    print(f"equivariant space rank: {equivariant_space_rank()}")
    if not resample_W: print(f"True W:\n{W}\n")
    else: print("True W: resampled every epoch (resample_W=True)")
    print(f"Initial Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Initial Ghat Lie generators:\n{Ghat.lie_algebra}")
    print()

    print(f"Dataset size: {dataset_size}")
    print(f"Effective dataset size (size * epochs): {epochs * dataset_size}")
    if resample_data < np.inf:
        print(f"Input datapoints are resampled every {resample_data} epochs")
    if resample_W < np.inf:
        print(f"True W matrix is resampled every {resample_W} epochs")
        if reset_model_on_resample: 
            print(f"Note: reset_model_on_resample=True, so the resamples of W trigger model weight re-initialization")

    ##### Training loop #####

    losses = defaultdict(list)
    batch_idx = 0 # time index of current training batch, for book-keeping
    for epoch in trange(epochs, desc="epochs"):
        if epoch % resample_data == 0: 
            X = np.random.normal(size=(dataset_size, repin(Ghat).size()))
        else:
            np.random.shuffle(X)

        if epoch % resample_W == 0:
            W = get_equivariant_W()
            f = map_from_matrix(W)
            if reset_model_on_resample:
                weight_vars.assign([objax.random.normal(var.shape) for var in weight_vars.tensors()])
    
        for batch_type in batch_types:
            for i in trange(num_batches[batch_type], desc=f"{batch_type} batches", leave=False):
                x = X[batch_size * i : batch_size * (i+1)]
                y = f(x)
                _, training_losses = train_ops[batch_type](x, y, lr)

                # Update losses dictionary
                for loss_name, loss_value in training_losses.items():
                    losses[loss_name].append((batch_idx, loss_value))

                batch_idx += 1

        x_val = np.random.normal(size=(num_val_batches * batch_size, repin(Ghat).size()))
        y_val = f(x_val)
        losses["/prediction/val"].append((batch_idx, loss(x_val, y_val)[1]["/prediction/train"]))
    print()

    ##### Plot info #####
    print_layer_info(model, Ghat)
    print()

    print(f"True W:\n{W}\n")
    print(f"W Ghat equivariance loss: {equivariance_loss(Ghat, repin, repout, W, ord=ord)}")
    print()

    print(f"Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Ghat Lie generators:\n{Ghat.lie_algebra}")

    plot_info(losses, len(model.network))
    assert False # for debugger

def plot_info(losses, num_layers):

    prefixes = {name.split('/')[0] for name in losses}
    non_layer_prefixes = {prefix for prefix in prefixes if "layer" not in prefix}
    prefix_to_row = {prefix: i for i,prefix in enumerate(non_layer_prefixes)}
    prefix_to_row.update( {f"layer {l}": l+len(non_layer_prefixes) for l in range(num_layers)} )
    type_to_col = {loss_type: idx for idx,loss_type in enumerate({name.split('/')[1] for name in losses})}
    
    fig, axes = plt.subplots(len(prefix_to_row), len(type_to_col))

    for loss_name, loss_list in losses.items():
        prefix, loss_type, label = loss_name.split('/')
        row = prefix_to_row[prefix]
        col = type_to_col[loss_type]
        ax = axes[row][col]
        xs, ys = zip(*loss_list)
        ax.plot(xs, ys, label=label, alpha=0.8)
        ax.legend()
        if row == 0: ax.set_title(loss_type)
        if col == 0: ax.set_ylabel(prefix)

    plt.show()

def print_layer_info(model, Ghat, print_reps=False):
    print("########## Layer info ##########")
    L = len(model.network)
    for l in range(L):
        print(f"===== Layer {l} =====")
        linear = model.network[l].linear if l < L - 1 else model.network[l]
        print(f"layer Ghat equivariance loss = {linear.equivariance_loss(Ghat, ord=ord)}")
        print(f"layer G equivariance loss = {linear.equivariance_loss(G, ord=ord)}")
        if reg_proj > 0:
            print(f"projection loss = {linear.projection_loss()}")
        print()

        print(f"repin: {linear.repin}")
        if print_reps:
            for h in Ghat.discrete_generators:
                print(f"discrete generator rep:\n{linear.repin.rho_dense(h)}")
            for A in Ghat.lie_algebra:
                print(f"Lie generator rep:\n{linear.repin.drho_dense(A)}")
            print()

        print(f"repout: {linear.repout}")
        if print_reps:
            for h in Ghat.discrete_generators:
                print(f"discrete generator rep:\n{linear.repout.rho_dense(h)}")
            for A in Ghat.lie_algebra:
                print(f"Lie generator rep:\n{linear.repout.drho_dense(A)}")
        print()

        print(f"b:\n{linear.b}")
        Ghat_err = equivariance_loss(Ghat, linear.repin, linear.repout, b=linear.b, ord=ord)
        print(f"bhat Ghat equivariance loss: {Ghat_err}")
        G_err = equivariance_loss(G, linear.repin, linear.repout, b=linear.b, ord=ord)
        print(f"bhat G equivariance loss: {G_err}")
        print()

        print(f"rep_W: {linear.rep_W}")
        if print_reps:
            for h in Ghat.discrete_generators:
                print(f"discrete generator rep:\n{linear.rep_W.rho_dense(h)}")
            for A in Ghat.lie_algebra:
                print(f"Lie generator rep:\n{linear.rep_W.drho_dense(A)}")
        print()

        print(f"W: {linear.repin} to {linear.repout}\n{linear.W}")
        print()

        Ghat_err = equivariance_loss(Ghat, linear.repin, linear.repout, linear.W, ord=ord)
        print(f"What Ghat equivariance loss: {Ghat_err}")
        G_err = equivariance_loss(G, linear.repin, linear.repout, linear.W, ord=ord)
        print(f"What G equivariance loss: {G_err}")
        print()

    print("################################")


if __name__ == "__main__":
    main()
