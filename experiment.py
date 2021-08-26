#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

import emlp
from emlp.learned_group import LearnedGroup
from emlp.groups import S, SO, O
from emlp.reps import V, equivariance_error
import emlp.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import objax
import jax.numpy as jnp
from jax import profiler
from tqdm.auto import tqdm

from jax.config import config
# For tracing where NaNs come from
config.update("jax_debug_nans", True)
# For examining values inside functions
config.update('jax_disable_jit', False)

alpha = 0.0 # regularization parameter: how much to weight equivariance loss 
beta = 0.0  # regularization parameter: how much to weight generator loss
lr = 8e-4
epochs = 10000
batch_size = 64

# Order of the matrix norm to use in loses
ord=2

n=2
G = S(n)
ncontinuous = len(G.lie_algebra)
ndiscrete = len(G.discrete_generators)
repin = V(G)
repout = V(G)
Proj = (repin >> repout).equivariant_projector()
W = np.random.normal(size=(n,n))
W = (Proj @ W.reshape(-1)).reshape(W.shape)

num_layers = 1
channels = 2

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
            loss += (jnp.linalg.norm(repin.drho_dense(A), ord=ord) - 1)**2
            loss += (jnp.linalg.norm(repout.drho_dense(A), ord=ord) - 1)**2
        return loss

def main():

    Ghat = LearnedGroup(n,ncontinuous,ndiscrete)
    #Ghat = G
    ngenerators = ncontinuous + ndiscrete
    model = nn.EMLP(repin, repout, LinearLayer=nn.ProjectionRecomputingLinear, group=Ghat, num_layers=num_layers, ch=channels)
    print(f"model vars:\n{model.vars()}")

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x, y):
        yhat = model(x)
        equivariance_loss = 0
        L = len(model.network)
        for l, layer in enumerate(model.network):
            if l < L - 1:
                equivariance_loss += layer.linear.equivariance_loss(Ghat, ord) / (L * ngenerators)
            else:
                equivariance_loss += layer.equivariance_loss(Ghat, ord) / (L * ngenerators)
        model_loss = ((yhat-y)**2).mean()
        g_loss = generator_loss(Ghat, repin, repout, ord) / ngenerators
        return (1-alpha-beta)*model_loss + alpha*equivariance_loss + beta*g_loss, \
                model_loss, equivariance_loss, g_loss

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v

    print(f"True W:\n{W}")
    print(f"Initial Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Initial Ghat Lie generators:\n{Ghat.lie_algebra}")



    model_losses = []
    equivariance_losses = []
    generator_losses = []
    equivariance_errors_learned = []
    equivariance_errors_true = []
    for epoch in tqdm(range(epochs)):
        x = np.random.normal(size=(batch_size, n)).squeeze()
        x_max = x.max(axis=-1)
        y = (W @ x[..., jnp.newaxis]).squeeze() + x_max[:, jnp.newaxis]
        loss, model_loss, equivariance_loss, g_loss = train_op(x, y, lr)

        model_losses.append(model_loss)
        equivariance_losses.append(equivariance_loss)
        generator_losses.append(g_loss)

        # Get equivariance error for first layer in network
        #linear = model.network[0].linear
        #equivariance_errors_learned.append(equivariance_error(linear.w, linear.repin(Ghat), linear.repout(Ghat), Ghat))
        #equivariance_errors_true.append(equivariance_error(linear.w, linear.repin(G), linear.repout(G), G))

    #fig, ((ax_model_loss, ax_null_loss), (ax_learned, ax_true)) = plt.subplots(2,2)
    fig, (ax_model_loss, ax_equivariance_loss, ax_generator_loss) = plt.subplots(1,3)
    
    ax_model_loss.plot(np.arange(epochs), model_losses)
    ax_model_loss.set_title("Model loss")

    ax_equivariance_loss.plot(np.arange(epochs), equivariance_losses)
    ax_equivariance_loss.set_title("Equivariance loss")

    ax_generator_loss.plot(np.arange(epochs), generator_losses)
    ax_generator_loss.set_title("Generator loss")

    #ax_learned.plot(np.arange(epochs), equivariance_errors_learned)
    #ax_learned.set_title("Leanred equivariance error")

    #ax_true.plot(np.arange(epochs), equivariance_errors_true)
    #ax_true.set_title("True equivariance error")
    
    plt.show()

    for l in range(num_layers+1):
        print(f"===== Layer {l} =====")
        linear = model.network[l].linear if l < num_layers else model.network[l]
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

        print(f"W: {linear.repin} to {linear.repout}\n{linear.w.value}")
        err = equivariance_error(linear.w.value, linear.repin(Ghat), linear.repout(Ghat), Ghat)
        print(f"equivariance error: {err}")

        print()

    print(f"Ghat discrete generators:\n{Ghat.discrete_generators}")
    print(f"Ghat Lie generators:\n{Ghat.lie_algebra}")

if __name__ == "__main__":
    main()
