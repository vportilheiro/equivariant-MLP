#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

# Logging must be set up before any JAX imports
import logging
logging.basicConfig(filename='experiment.log', encoding='utf-8', level=logging.DEBUG)

import emlp
from emlp.learned_group import LearnedGroup, equivariance_loss
from emlp.groups import S, SO, O, Trivial
from emlp.reps import V, equivariance_error
import emlp.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import objax
import jax
import jax.numpy as jnp
from jax.lax import dynamic_update_slice
from tqdm.auto import tqdm

##### Debuggin options #####

from jax.config import config
config.update("jax_debug_nans", False) # For tracing where NaNs come from
config.update('jax_disable_jit', False) # For examining values inside functions

##### Training hyperparameters #####

alpha = 0.0 # regularization parameter: how much to weight equivariance loss 
beta = 0.0  # regularization parameter: how much to weight generator loss
gamma = 0.9  # regularization parameter: how much to weight projection loss
lr = 1e-4
epochs = 20000
batch_size = 64

# Order of the matrix norm to use in losses
ord=2

# Model parameters
num_layers = 1
channels = V**0 + V

##### Task setup #####

n = 3
G = S(n)
ncontinuous = len(G.lie_algebra)
ndiscrete = len(G.discrete_generators)
repin = V(G)
repout = V(G)

def get_equivariant_W():
    Proj = (repin >> repout).equivariant_projector()
    W = np.random.normal(size=(n,n))
    W = (Proj @ W.reshape(-1)).reshape(W.shape)
    return W

def W_map(W):
    return lambda x: (W @ x[..., jnp.newaxis]).squeeze()

W = get_equivariant_W()
f = W_map(W)
 
def generator_loss(G, repin, repout, ord=2):
        repin = repin(G)
        repout = repout(G)
        loss = 0
        for h in G.discrete_generators:
            H_in = repin.rho_dense(h)
            H_out = repout.rho_dense(h)
            loss += jnp.linalg.norm(H_in, ord=ord)
            loss += jnp.linalg.norm(H_out, ord=ord)
            # Penalizes being close to the identity
            #loss -= jnp.linalg.norm(H_in - jnp.eye(H_in.shape[-1]), ord=ord)
            #loss -= jnp.linalg.norm(H_out - jnp.eye(H_out.shape[-1]), ord=ord)
        for A in G.lie_algebra:
            loss += jnp.linalg.norm(repin.drho_dense(A), ord=ord)
            loss += jnp.linalg.norm(repout.drho_dense(A), ord=ord)
        return loss

def main():

    Ghat = LearnedGroup(n,ncontinuous,ndiscrete)
    #Ghat = G
    #Ghat = Trivial(n)
    ngenerators = ncontinuous + ndiscrete

    class RankOneLinear(nn.ProjectionRecomputingLinear):
        def __init__(self, repin, repout):
            super().__init__(repin, repout, lambda S: dynamic_update_slice(jnp.zeros_like(S), jnp.ones(1), [-1]))
        def projection_loss(self):
            return sum((S*mask) @ (S*mask) for (S,mask) in self.sv_w_dict.values())

    model = nn.EMLP(repin, repout, LinearLayer=RankOneLinear, group=Ghat, num_layers=num_layers, ch=channels)
    print(f"model vars:\n{model.vars()}\n")

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x, y):
        yhat = model(x)
        equivariance_loss = 0
        projection_loss = 0
        L = len(model.network)
        for l, layer in enumerate(model.network):
            if l < L - 1:
                equivariance_loss += layer.linear.equivariance_loss(Ghat, ord) / (L * ngenerators)
                if gamma > 0:
                    projection_loss += layer.linear.projection_loss() / L
            else:
                equivariance_loss += layer.equivariance_loss(Ghat, ord) / (L * ngenerators)
                if gamma > 0:
                    projection_loss += layer.projection_loss() / L
        model_loss = ((yhat-y)**2).mean()
        g_loss = 0
        g_loss += generator_loss(Ghat, repin, repout, ord) / ngenerators
        return (1-alpha-beta-gamma)*model_loss + alpha*equivariance_loss + beta*g_loss + gamma * projection_loss, \
                model_loss, equivariance_loss, g_loss, projection_loss

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v

    #print(f"True W:\n{W}")
    print(f"Initial Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Initial Ghat Lie generators:\n{Ghat.lie_algebra}")

    model_losses = []
    equivariance_losses = []
    generator_losses = []
    projection_losses = []
    for epoch in tqdm(range(epochs)):
        x = np.random.normal(size=(batch_size, repin.size())).squeeze()
        y = f(x)
        loss, model_loss, equivariance_loss, g_loss, proj_loss = train_op(x, y, lr)

        model_losses.append(model_loss)
        equivariance_losses.append(equivariance_loss)
        generator_losses.append(g_loss)
        projection_losses.append(proj_loss)

        # Get equivariance error for first layer in network
        #linear = model.network[0].linear
        #equivariance_errors_learned.append(equivariance_error(linear.w, linear.repin(Ghat), linear.repout(Ghat), Ghat))
        #equivariance_errors_true.append(equivariance_error(linear.w, linear.repin(G), linear.repout(G), G))

    #fig, ((ax_model_loss, ax_null_loss), (ax_learned, ax_true)) = plt.subplots(2,2)
    fig, (ax_model_loss, ax_equivariance_loss, ax_generator_loss, ax_proj_loss) = plt.subplots(1,4)
    
    ax_model_loss.plot(np.arange(epochs), model_losses)
    ax_model_loss.set_title("Model loss")

    ax_equivariance_loss.plot(np.arange(epochs), equivariance_losses)
    ax_equivariance_loss.set_title("Equivariance loss")

    ax_generator_loss.plot(np.arange(epochs), generator_losses)
    ax_generator_loss.set_title("Generator loss")

    ax_proj_loss.plot(np.arange(epochs), projection_losses)
    ax_proj_loss.set_title("Projection loss")

    #ax_learned.plot(np.arange(epochs), equivariance_errors_learned)
    #ax_learned.set_title("Leanred equivariance error")

    #ax_true.plot(np.arange(epochs), equivariance_errors_true)
    #ax_true.set_title("True equivariance error")
    
    plt.show()

    print_layer_info(model, Ghat)

    print(f"Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Ghat Lie generators:\n{Ghat.lie_algebra}")
    

def print_layer_info(model, Ghat):
    for l in range(num_layers+1):
        print(f"===== Layer {l} =====")
        linear = model.network[l].linear if l < num_layers else model.network[l]
        print(f"layer Ghat equivariance loss = {linear.equivariance_loss(Ghat)}")
        print(f"layer G equivariance loss = {linear.equivariance_loss(G)}")
        #print(f"projection loss = {linear.projection_loss()}")
        print()

        print(f"repin: {linear.repin}")
        for h in Ghat.discrete_generators:
            print(f"discrete generator rep:\n{linear.repin(G).rho_dense(h)}")
        for A in Ghat.lie_algebra:
            print(f"Lie generator rep:\n{linear.repin(G).drho_dense(A)}")
        print()

        print(f"repout: {linear.repout}")
        for h in Ghat.discrete_generators:
            print(f"discrete generator rep:\n{linear.repout(G).rho_dense(h)}")
        for A in Ghat.lie_algebra:
            print(f"Lie generator rep:\n{linear.repout(G).drho_dense(A)}")
        print()

        print(f"b:\n{linear.b}")
        print()

        print(f"rep_W: {linear.rep_W}")
        for h in Ghat.discrete_generators:
            print(f"discrete generator rep:\n{linear.rep_W(G).rho_dense(h)}")
        for A in Ghat.lie_algebra:
            print(f"Lie generator rep:\n{linear.rep_W(G).drho_dense(A)}")
        print()

        print(f"W: {linear.repin} to {linear.repout}\n{linear.W}")
        print()

        Ghat_err = equivariance_error(linear.W, linear.repin(Ghat), linear.repout(Ghat), Ghat)
        print(f"What Ghat equivariance error: {Ghat_err}")
        G_err = equivariance_error(linear.W, linear.repin(G), linear.repout(G), G)
        print(f"What G equivariance error: {G_err}")
        print()

    Ghat_err = equivariance_error(W, repin(Ghat), repout(Ghat), Ghat)
    print(f"W Ghat equivariance error: {Ghat_err}")
    G_err = equivariance_error(W, repin(G), repout(G), G)
    print(f"W G equivariance error: {G_err}")
    print(f"W Ghat equivariance loss: {equivariance_loss(Ghat, repin, repout, W)}")
    print(f"W G equivariance loss: {equivariance_loss(G, repin, repout, W)}")
    print()


if __name__ == "__main__":
    main()
