# %%
import torch
from torch import nn
#%% ------------ Return g(X) --------------
def g_D(X_test, Z_train,X_train,De_train,Lambda_U,Beta0,n_layer,n_node,n_lr,n_epoch):
    X_test = torch.Tensor(X_test)
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    De_train = torch.Tensor(De_train)
    Lambda_U = torch.Tensor(Lambda_U)
    Beta0 = torch.Tensor(Beta0)
    # -------------- Define network structure  --------------
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(4, n_node))
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


    # -------------- Preparation before training  --------------
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De, Z, Beta, Lambda_U, g_X):
        # Ensure that the shape of g_X is consistent with that of De, Z, U
        assert(len(g_X.shape) == len(De.shape))
        assert(len(g_X.shape) == len(Lambda_U.shape))
        Lam1 = Lambda_U * torch.exp(Z[:,0]*Beta[0] +Z[:,1]*Beta[1] + g_X)
        loss_fun = -torch.mean(De*torch.log(1-torch.exp(-Lam1)+1e-5) - (1-De)*Lam1) # Add 1e-5 to the log to prevent it from becoming nan
        return loss_fun


    # ------------ training  -----------------
    for epoch in range(n_epoch):
        pred_g_X = model(X_train) # The result predicted by the model is a matrix (n*1), which needs to be converted into a vector later
        loss = my_loss(De_train, Z_train, Beta0, Lambda_U, pred_g_X[:, 0])
        # print('epoch=', epoch, 'loss=', loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    #%% ----------- complete training ------------
    # forecast
    g_train = model(X_train)
    g_train = g_train[:,0].detach().numpy() # The default output of the neural network is in the form of a matrix n*1 matrix, which needs to be converted into a vector
    g_test = model(X_test)
    g_test = g_test[:,0].detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test
    }