#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

import emlp
from emlp.learned_group import LearnedGroup
from emlp.groups import S
from emlp.reps import V, equivariance_error
import emlp.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import objax
import jax.numpy as jnp
from tqdm.auto import tqdm

# For tracing where NaNs come from
#from jax.config import config
#config.update("jax_debug_nans", True)

alpha = 0.5 # regularization parameter: how much to weight equivariance loss 
beta = 0.0  # regularization parameter: how much to weight generator loss
lr = 8e-4
epochs = 20000 
batch_size = 64

# Order of the matrix norm to use in loses
ord=2

n=2
W = 3*jnp.eye(n) + 2*jnp.ones((n,n))

num_layers = 1
channels = 2

def main():

    G = S(n)
    Ghat = LearnedGroup(n,ncontinuous=0,ndiscrete=n-1)
    ngenerators = n-1
    repin = V
    repout = V
    model = nn.EMLP(repin, repout, LinearLayer=nn.ApproximatingLinear, group=Ghat, num_layers=num_layers, ch=channels)

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x, y):
        yhat = model(x)
        equivariance_loss = 0
        generator_loss = 0
        L = len(model.network)
        for l, layer in enumerate(model.network):
            if l < L - 1:
                equivariance_loss += layer.linear.equivariance_loss(Ghat, ord) / (L * ngenerators)
                generator_loss += layer.linear.generator_loss(Ghat, ord) / (L * ngenerators)
            else:
                equivariance_loss += layer.equivariance_loss(Ghat, ord) / (L * ngenerators)
                generator_loss += layer.generator_loss(Ghat, ord) / (L * ngenerators)
        model_loss = ((yhat-y)**2).mean()
        return (1-alpha-beta)*model_loss + alpha*equivariance_loss + beta*generator_loss, \
                model_loss, equivariance_loss, generator_loss

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v

    model_losses = []
    equivariance_losses = []
    generator_losses = []
    equivariance_errors_learned = []
    equivariance_errors_true = []
    for epoch in tqdm(range(epochs)):
        x = objax.random.normal((batch_size, n)).squeeze()
        x_max = x.max(axis=-1)
        y = (W @ x[..., jnp.newaxis]).squeeze() + x_max[:, jnp.newaxis]
        _, model_loss, equivariance_loss, generator_loss = train_op(x, y, lr)

        model_losses.append(model_loss)
        equivariance_losses.append(equivariance_loss)
        generator_losses.append(generator_loss)

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

        print()

    print(f"Ghat discrete generators: {Ghat.discrete_generators}")
    print(f"Ghat Lie generators: {Ghat.lie_algebra}")

if __name__ == "__main__":
    main()
