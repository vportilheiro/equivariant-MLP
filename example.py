import emlp
from emlp.learned_group import LearnedGroup
from emlp.groups import S
from emlp.reps import V, equivariance_error

import objax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

alpha = 0.8 # regularization parameter: how much to weight null space loss vs model loss
lr = 5e-5
epochs = 50000#250000
batch_size = 64

n=3
W = 3*np.eye(n) + 2*np.ones((n,n))

num_layers = 1

def main():

    G = S(n)
    Ghat = LearnedGroup(n,ncontinuous=0,ndiscrete=n-1)
    repin = V(2)
    repout = V(1)
    model = emlp.nn.LearnedGroupEMLP(repin, repout, group=Ghat, num_layers=num_layers, ch=2)

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x,y,alpha=alpha):
        model_loss = ((model(x) - y)**2).mean()
        null_space_loss = model.null_space_loss()
        return (1-alpha)*model_loss + alpha*null_space_loss, model_loss, null_space_loss

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v

    model_losses = []
    null_space_losses = []
    equivariance_errors_learned = []
    equivariance_errors_true = []
    for epoch in tqdm(range(epochs)):
        x = np.random.rand(batch_size, n).squeeze()
        y = (W @ x[..., np.newaxis]).squeeze()
        total_loss, model_loss, null_space_loss = train_op(jnp.array(x),jnp.array(y),lr)
        model_losses.append(model_loss)
        null_space_losses.append(null_space_loss)

        # Get equivariance error for first layer in network
        #linear = model.network[0].linear
        #equivariance_errors_learned.append(equivariance_error(linear.w, linear.repin(Ghat), linear.repout(Ghat), Ghat))
        #equivariance_errors_true.append(equivariance_error(linear.w, linear.repin(G), linear.repout(G), G))

    fig, ((ax_model_loss, ax_null_loss), (ax_learned, ax_true)) = plt.subplots(2,2)
    
    ax_model_loss.plot(np.arange(epochs), model_losses)
    ax_model_loss.set_title("Model loss")

    ax_null_loss.plot(np.arange(epochs), null_space_losses)
    ax_null_loss.set_title("Null space loss")

    #ax_learned.plot(np.arange(epochs), equivariance_errors_learned)
    #ax_learned.set_title("Leanred equivariance error")

    #ax_true.plot(np.arange(epochs), equivariance_errors_true)
    #ax_true.set_title("True equivariance error")
    
    plt.show()

    for l in range(num_layers+1):
        print(f"===== Layer {l} =====")
        linear = model.network[l].linear if l < num_layers else model.network[l]
        Q, loss = (linear.rep_W).equivariant_basis()
        print(f"Q = {Q.to_dense()}")

        Proj, _ = (linear.rep_W).equivariant_projector()
        print(f"Proj = {Proj.to_dense()}")

        print(f"loss = {loss}")
        print()

    print(f"Ghat generators: {Ghat.discrete_generators}")

if __name__ == "__main__":
    main()
