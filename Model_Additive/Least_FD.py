# %%
import torch
from torch import nn
import numpy as np

#%% ------------ Return a(U)+b(X) --------------
def LFD(train_data,Lambda_U,g_train,Beta,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    De_train = torch.Tensor(train_data['De'])
    X_U = torch.Tensor(np.c_[train_data['X'], train_data['U']])
    Lambda_U = torch.Tensor(Lambda_U)
    Beta = torch.Tensor(np.array([Beta]))
    # -------------- Define network structure --------------
    class DNNAB(torch.nn.Module):
        def __init__(self):
            super(DNNAB, self).__init__()
            layers = []
            layers.append(nn.Linear(6, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 1))
            self.model = nn.Sequential(*layers)
            # Here nn.Linear is a linear combination, and nn.ReLU() is an activation function, so the fronts appear in pairs
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred


    # -------------- Preparation before training --------------
    model = DNNAB()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def Loss(De, Z, Beta, Lambda_U, g_X, a_b):
        # Ensure that the shape of a_b is consistent with that of De, Z, U
        assert(len(a_b.shape) == len(De.shape))
        assert(len(a_b.shape) == len(Z.shape))
        assert(len(a_b.shape) == len(Lambda_U.shape))
        h_v = Lambda_U * torch.exp( Z*Beta + g_X)
        Q_y = h_v * (De * torch.exp(-h_v)/(1-torch.exp(-h_v)+1e-5) - (1-De))
        Loss_f = torch.mean(Q_y**2 * (Z-a_b)**2)
        return Loss_f


    # ------------ training -----------------
    for epoch in range(n_epoch):
        pred_ab = model(X_U)  # The result predicted by the model is a matrix (n*1), which needs to be converted into a vector later
        loss = Loss(De_train, Z_train, Beta, Lambda_U, g_train, pred_ab[:, 0])
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#%% ----------- complete training  ------------
    ab_train = model(X_U)
    ab_train = ab_train[:,0].detach().numpy()
    return ab_train