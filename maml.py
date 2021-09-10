#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

import emlp
from emlp.learned_group import LearnedGroup, equivariance_loss, generator_loss
from emlp.groups import S, SO, O
from emlp.reps import V, equivariance_error
import emlp.nn as nn

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import objax
import jax
import jax.numpy as jnp
from jax.lax import dynamic_update_slice
from tqdm.auto import tqdm

from jax.config import config
# For tracing where NaNs come from
config.update("jax_debug_nans", False)
# For examining values inside functions
config.update('jax_disable_jit', False)

reg_eq = 0.50 # regularization parameter: how much to weight equivariance loss 
reg_gen = 0.0 #1e-8  # regularization parameter: how much to weight generator loss
reg_proj = 0.0  # regularization parameter: how much to weight projection loss
reg_sv = 0.0 # how much to weight loss from singular values

outer_epochs = 10000
inner_epochs = 10
inner_batch_size = 64
outer_batch_size = 64
inner_lr = 1e-4
outer_lr = 1e-4

# Order of the matrix norm to use in loses
ord=2

n=3
G = S(n)
ncontinuous = len(G.lie_algebra)
ndiscrete = len(G.discrete_generators)
repin = V(G)
repout = V(G)
Proj = (repin >> repout).equivariant_projector()

def sample_tasks(inner_batch_size, outer_batch_size, return_Ws=False):
    Ws = np.random.normal(size=(outer_batch_size,n,n))
    Ws = (Proj @ Ws.reshape(outer_batch_size, -1).T).T.reshape(Ws.shape)

    def get_batch():
        Xs = np.random.normal(size=(outer_batch_size,inner_batch_size,n))
        Ys = (Ws[:, np.newaxis] @ Xs[..., np.newaxis]).squeeze(-1)
        return Xs, Ys

    Xs_train, Ys_train = get_batch()
    Xs_val, Ys_val = get_batch()
    if return_Ws:
        return Xs_train, Ys_train, Xs_val, Ys_val, Ws
    return Xs_train, Ys_train, Xs_val, Ys_val

num_layers = 1
channels = V**0 + V # NOTE: the order of summation matters here for implementation reasons (probably a bug)

def main():

    Ghat = LearnedGroup(n,ncontinuous,ndiscrete)
    #Ghat = G
    ngenerators = ncontinuous + ndiscrete

    model = nn.EMLP(repin, repout, \
            #LinearLayer=nn.SoftSVDLinear(1, sv_loss_func=lambda S: jnp.sum(jnp.tanh(S/5))), \
            LinearLayer=nn.ApproximatingLinear,
            group=Ghat, num_layers=num_layers, ch=channels)

    all_vars = model.vars()
    generator_vars = objax.VarCollection((k,v) for k,v in all_vars.items() if "LearnedGroup" in k)
    weight_vars = objax.VarCollection((k,v) for k,v in all_vars.items() if "LearnedGroup" not in k)

    print(f"group vars:\n{generator_vars}")
    print(f"non-group vars:\n{weight_vars}")

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    #@objax.Function.with_vars(model.vars())
    @objax.Function.with_vars(weight_vars)
    def inner_loss(X, Y):
        Y_pred = model(X)
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

        losses["/prediction/train"] = ((Y_pred-Y)**2).mean()
        losses["/norm/generators"] = generator_loss(Ghat, ord) / ngenerators
        p = losses["/svd/proj"] if reg_proj > 0 else 0
        s = losses["/svd/sv"] if reg_sv > 0 else 0
        m, e, g = losses["/prediction/train"], losses["/equivariance/model-Ghat"], \
                     losses["/norm/generators"]
        return ((1-reg_eq-reg_gen-reg_proj-reg_sv)*m+ reg_eq*e + reg_gen*g + reg_proj*p + reg_sv*s), losses

    inner_grad_and_val = objax.GradValues(inner_loss, weight_vars)

    @objax.Function.with_vars(objax.VarCollection())
    def no_gradient_update(X, Y, num_steps):
        for _ in tqdm(range(num_steps), desc='inner loop', leave=False):
            inner_grad = inner_grad_and_val(X, Y)[0] 
            weight_vars.assign([weight - inner_lr * grad for weight,grad in zip(weight_vars.tensors(), inner_grad)])

    @objax.Function.with_vars(model.vars())
    #@objax.Function.with_vars(generator_vars)
    def outer_loss(X_train, Y_train, X_val, Y_val, W, inner_lr):
        # Save model weights before inner task training
        original_weights = weight_vars.tensors() 
        # Obtain model weights by training on inner task
        for _ in tqdm(range(inner_epochs), desc='inner loop', leave=False):
            inner_grad = inner_grad_and_val(X_train, Y_train)[0] 
            weight_vars.assign([weight - inner_lr * grad for weight,grad in zip(weight_vars.tensors(), inner_grad)])
        #no_gradient_update(X_train, Y_train, inner_epochs)
        # Return the loss obtained with these new weights, but restore pre-inner-task weights
        loss, losses = inner_loss(X_val, Y_val)
        weight_vars.assign(original_weights)
        # Calculate equivariance loss of W with respect to Ghat (does not change over inner loop)
        losses["/equivariance/W-Ghat"] = Ghat_equivariance(W)
        return loss, losses

    vec_outer_loss = objax.Vectorize(outer_loss, batch_axis=(0,0,0,0,0,None))

    @objax.Function.with_vars(vec_outer_loss.vars())
    def batch_outer_loss(Xs_train, Ys_train, Xs_val, Ys_val, Ws, inner_lr):
        loss, losses = \
                vec_outer_loss(Xs_train, Ys_train, Xs_val, Ys_val, Ws, inner_lr)
        return loss.mean(), {name: l.mean() for name, l in losses.items()}

    outer_grad_and_val = objax.GradValues(batch_outer_loss, vec_outer_loss.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(Xs_train, Ys_train, Xs_val, Ys_val, Ws, inner_lr, outer_lr):
        g, v = outer_grad_and_val(Xs_train, Ys_train, Xs_val, Ys_val, Ws, inner_lr)
        opt(lr=outer_lr, grads=g)
        return v

    @objax.Jit
    @objax.Function.with_vars(Ghat.vars())
    def Ghat_equivariance(W):
        return equivariance_loss(Ghat, repin, repout, W, ord=ord)


    print(f"Initial Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Initial Ghat Lie generators:\n{Ghat.lie_algebra}")

    losses = defaultdict(list)
    for epoch in tqdm(range(outer_epochs),desc='outer loop'):
        Xs_train, Ys_train, Xs_val, Ys_val, Ws = sample_tasks(inner_batch_size, outer_batch_size, return_Ws=True)
        loss, training_losses = \
                train_op(Xs_train, Ys_train, Xs_val, Ys_val, Ws, inner_lr, outer_lr)

        for loss_name, loss_value in training_losses.items():
            losses[loss_name].append((epoch, loss_value))

    ##### Plot info #####
    print_layer_info(model, Ghat)
    print()

    print(f"Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Ghat Lie generators:\n{Ghat.lie_algebra}")

    plot_info(losses, len(model.network))

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
