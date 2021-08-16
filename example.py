#####
# This file contains a simple experiment for learning groups using the "null space loss".
# The map we are trying to learn --- the linear map W --- is made to be equivariant under
# the symmetric group.
#####

import emlp
from emlp.learned_group import LearnedGroup
from emlp.groups import S
from emlp.reps import V, equivariance_error
import emlp.nn.pytorch as nn

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

alpha = 0.8 # regularization parameter: how much to weight null space loss vs model loss
lr = 8e-4
epochs = 10000 
batch_size = 64

n=3
W = 3*torch.eye(n) + 2*torch.ones((n,n))

num_layers = 1

def main():

    G = S(n)
    Ghat = LearnedGroup(n,ncontinuous=0,ndiscrete=n-1)
    repin = V(1)
    repout = V(1)
    model = nn.LearnedGroupEMLP(repin, repout, group=Ghat, num_layers=num_layers, ch=2)

    opt = torch.optim.Adam(model.parameters(),lr=lr)

    model_losses = []
    null_space_losses = []
    equivariance_errors_learned = []
    equivariance_errors_true = []
    for epoch in tqdm(range(epochs)):
        opt.zero_grad()
        x = torch.normal(0,1,size=(batch_size, n)).squeeze()
        y = (W @ x[..., np.newaxis]).squeeze()
        yhat = model(x)

        model_loss = ((model(x) - y)**2).mean()
        null_space_loss = model.null_space_loss()
        loss = (1-alpha)*model_loss + alpha*null_space_loss
        loss.backward()
        opt.step()

        model_losses.append(model_loss)
        null_space_losses.append(null_space_loss)

        # Get equivariance error for first layer in network
        #linear = model.network[0].linear
        #equivariance_errors_learned.append(equivariance_error(linear.w, linear.repin(Ghat), linear.repout(Ghat), Ghat))
        #equivariance_errors_true.append(equivariance_error(linear.w, linear.repin(G), linear.repout(G), G))

    #fig, ((ax_model_loss, ax_null_loss), (ax_learned, ax_true)) = plt.subplots(2,2)
    fig, (ax_model_loss, ax_null_loss) = plt.subplots(1,2)
    
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
        print(f"rep W: {linear.rep_W}")

        Q, loss = (linear.rep_W).equivariant_basis()
        print(f"Q = {Q.to_dense().detach().numpy()}")

        Proj, _ = (linear.rep_W).equivariant_projector()
        print(f"Proj = {Proj.to_dense().detach().numpy()}")

        print(f"null space loss = {loss}")

        print(f"Projected weight = {(linear.Pw @ linear.weight.reshape(-1)).reshape(linear.weight.shape).detach().numpy()}")

        print()


    print(f"Ghat generators: {Ghat.discrete_generators.detach().numpy()}")

if __name__ == "__main__":
    main()
