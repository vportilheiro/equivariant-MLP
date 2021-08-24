#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

import emlp
from emlp.learned_group import LearnedGroup
from emlp.groups import S,SO
from emlp.reps import V, equivariance_error
import emlp.nn.pytorch as nn

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

alpha = 0.0 # regularization parameter: how much to weight equivariance loss
beta = 0.0  # regularization parameter: how much to weight generator loss
gamma = 0.0 # regularization parameter: how much to weight null space loss
lr = 8e-4
epochs = 20000 
batch_size = 64

n=2
G = SO(n).torchify()

repin = V
repout = V
Proj, _ = (repin >> repout)(G).equivariant_projector()
W = torch.Tensor([[0.0,-1.0],[1.0,0.0]])
W = (Proj @ W.reshape(-1)).reshape(W.shape)
print(f"True W:\n {W}")

num_layers = 1
channels = [V+V**2]
W_ranks = [2]
b_ranks = [1]
ncontinuous = len(G.lie_algebra)
ndiscrete = len(G.discrete_generators)

def main():

    Ghat = LearnedGroup(n,ncontinuous,ndiscrete)
    model = nn.LearnedGroupEMLP(repin, repout, group=Ghat, W_ranks=W_ranks, b_ranks = b_ranks, num_layers=num_layers, ch=channels)

    opt = torch.optim.Adam(model.parameters(),lr=lr)

    model_losses = []
    null_space_losses = []
    equivariance_losses = []
    generator_losses = []
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        with torch.autograd.detect_anomaly():
            x = torch.normal(0,1,size=(batch_size, n)).squeeze()
            y = (W @ x[..., np.newaxis]).squeeze()
            yhat = model(x)

            model_loss = ((model(x) - y)**2).mean()
            equivariance_loss = model.equivariance_loss(Ghat)
            generator_loss = model.generator_loss(Ghat)
            null_space_loss = model.null_space_loss()
            loss = (1-alpha-beta-gamma)*model_loss + alpha*equivariance_loss + beta*generator_loss + gamma*null_space_loss
            if True:
                model_loss.retain_grad()
                equivariance_loss.retain_grad()
                generator_loss.retain_grad()
                null_space_loss.retain_grad()
                loss.backward()
                for l,name in [(model_loss, "m-"), (equivariance_loss, "eq-"), (generator_loss, "g-"), (null_space_loss, "n-")]:
                    print(f"{name}loss: {l}\n{name}loss grad: {l.grad}")
            opt.step()

        model_losses.append(model_loss)
        equivariance_losses.append(equivariance_loss)
        generator_losses.append(generator_loss)
        null_space_losses.append(null_space_loss)

    fig, (ax_model_loss, ax_equivariance_loss, ax_generator_loss, ax_null_space_loss) = plt.subplots(1,4)
    
    ax_model_loss.plot(np.arange(epochs), model_losses)
    ax_model_loss.set_title("Model loss")

    ax_equivariance_loss.plot(np.arange(epochs), equivariance_losses)
    ax_equivariance_loss.set_title("Equivariance loss")

    ax_generator_loss.plot(np.arange(epochs), generator_losses)
    ax_generator_loss.set_title("Generator loss")

    ax_null_space_loss.plot(np.arange(epochs), null_space_losses)
    ax_null_space_loss.set_title("Null space loss")

    plt.show()

    for l in range(num_layers+1):
        print(f"===== Layer {l} =====")
        linear = model.network[l].linear if l < num_layers else model.network[l]
        print(f"rep W: {linear.rep_W}")

        Q, loss = (linear.rep_W).equivariant_basis()
        print(f"Q = {Q.to_dense().detach().numpy()}")

        Proj, _ = (linear.rep_W).equivariant_projector()
        print(f"Proj = {Proj.to_dense().detach().numpy()}")

        print(f"Projected weight = {(linear.Pw @ linear.weight.reshape(-1)).reshape(linear.weight.shape).detach().numpy()}")

        print()


    print(f"Ghat generators: {Ghat.discrete_generators.detach().numpy()}")

if __name__ == "__main__":
    main()
