import emlp
from emlp.learned_group import LearnedGroup
from emlp.reps import V

import objax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

alpha = 0.5
lr = 1e-3
epochs = 250000

W = 3*np.eye(3) + 2*np.ones((3,3))

def main():

    G = LearnedGroup(3,ncontinuous=0,ndiscrete=2)
    model = emlp.nn.LearnedGroupEMLP(V(2), V(2), group=G, num_layers=2, ch=2)

    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss(x,y,alpha=0.5):
        return (1-alpha)*((model(x) - y)**2).mean() + alpha * model.null_space_loss()

    grad_and_val = objax.GradValues(loss, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v


    test_losses = []
    train_losses = []
    for epoch in tqdm(range(epochs)):
        x = np.random.rand(3)
        y = W @ x
        train_losses.append(train_op(jnp.array(x),jnp.array(y),lr))
        if not epoch%10:
            test_losses.append(loss(jnp.array(x),jnp.array(y)))

    plt.plot(np.arange(epochs),train_losses,label='Train loss')
    plt.plot(np.arange(0,epochs,10),test_losses,label='Test loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
