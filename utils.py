import numpy as np
import xarray as xr
from numba import jit
from pytorch_lightning import LightningModule, Trainer
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# Standard set of parameters is given in https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved
# For two-scale system it is F=10, h=1, b=10, c=10 and dt=0.005, K=36, J=10

def RK4(fn, dt, X, *kw):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the fourth order Runge-Kutta method.
    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The time step
        X      : Values of X variables at the current time, t
        kw     : All other arguments that should be passed to fn, i.e. fn(X, t, *kw)
    Returns:
        X at t+dt
    """

    Xdot1 = fn(X, *kw)
    Xdot2 = fn(X + 0.5 * dt * Xdot1, *kw)
    Xdot3 = fn(X + 0.5 * dt * Xdot2, *kw)
    Xdot4 = fn(X + dt * Xdot3, *kw)
    return X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))

@jit
def L96_rhs(X, Y, F=20, h=1, b=10, c=10):
    """
    Calculate the time rate of change for the X and Y variables for the Lorenz '96, two time-scale
    model, equations 2 and 3:
        d/dt X[k] =     -X[k-1] ( X[k-2] - X[k+1] )   - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] ( Y[j+2] - Y[j-1] ) - c Y[j]     + h.c/b X[k]
    Args:
        X : Values of X variables at the current time step
        Y : Values of Y variables at the current time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt, dYdt, C : Arrays of X and Y time tendencies, and the coupling term -hc/b*sum(Y,j)
    """
    #-------------------------------------------------------#
    Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F
    yield Xdot

    #-------------------------------------------------------#
    hcb = (h * c) / b

    JK, K = len(Y), len(X)
    J = JK // K
    assert JK == J * K, "X and Y have incompatible shapes"
    Ysummed = Y.reshape((K, J)).sum(axis=-1)
    
    Xcouple = - hcb * Ysummed
    yield Xcouple
    #-------------------------------------------------------#
    Ydot = (
        -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
        - c * Y
        + hcb * np.repeat(X, J)
    )
    yield Ydot

def L96_torch(X, F=20, h=1, b=10, c=10):
    '''
    Compared to upper function works on batches,
    i.e. dimension is Nbatch x K
    '''
    return torch.roll(X, 1, -1) * (torch.roll(X, -1, -1) - torch.roll(X, 2, -1)) - X + F

class L96():
    def __init__(self, K=8, J=32, F=20, h=1, b=10, c=10, dt=0.005):
        self.K = K
        self.J = J
        self.F = F
        self.h = h
        self.b = b
        self.c = c
        self.dt = dt
        self.random_init()
    def random_init(self):
        self.t = 0
        self.X = np.random.randn(self.K)
        self.Y = np.random.randn(self.K * self.J)
    def step(self):
        NotImplementedError
    def warmup(self, N=5000):
        for i in range(N):
            self.step()
    def run(self, N=5000, every=1, collect_coupling=False):
        def add_point():
            X.append(self.X)
            Y.append(self.Y)
            t.append(self.t)
            if collect_coupling:
                _, XX, _ = L96_rhs(self.X, self.Y, self.F, self.h, self.b, self.c)
                Xcouple.append(XX)

        X, Y, t, Xcouple = [], [], [], []

        add_point()
        for i in range(N-1):
            for j in range(every):
                self.step()
            add_point()

        ds = xr.Dataset()
        ds['t'] = xr.DataArray(t, dims=['t'])
        ds['k'] = xr.DataArray(np.linspace(0,1,self.K), dims=['k'])
        ds['jk'] = xr.DataArray(np.linspace(0,1,self.J*self.K), dims=['jk'])
        ds['X'] = xr.DataArray(X, dims=['t', 'k'])
        ds['Y'] = xr.DataArray(Y, dims=['t', 'jk'])
        if collect_coupling:
            ds['Xcouple'] = xr.DataArray(Xcouple, dims=['t', 'k'])

        return ds
    def init(self, m):
        self.t = m.t
        self.X = m.X
        self.Y = m.Y

class L96_X(L96):
    def __init__(self, sgs=lambda X: 0*X, **kw):
        super().__init__(**kw)
        self.sgs = sgs

    def step(self):
        # fetch only Xdot from the generator for a given X
        fn = lambda X: next(L96_rhs(X, self.Y, self.F, self.h, self.b, self.c)) + self.sgs(X)
        self.X = RK4(fn, self.dt, self.X)
        self.t = self.t + self.dt

    @classmethod
    def from_dataset(cls, _train_loader, _val_loader, ds_test, beta=0, gamma=0, max_epochs=100, train_kw={}):
        model = ANN(0.005, beta, gamma, 8)
        model.do_training(_train_loader, _val_loader, max_epochs=max_epochs, **train_kw)

        def sgs(X):
            _X = torch.tensor(X.astype(np.float32)).reshape(1,8)
            return model(_X).detach().numpy().squeeze()
        
        return cls(sgs=sgs), model.predict(ds_test)
    
    @classmethod
    def train_online(cls, _train_loader, _val_loader, ds_test, max_epochs=100, train_kw={}):
        model = ANN_online(0.005, 8)
        model.do_training(_train_loader, _val_loader, max_epochs=max_epochs, **train_kw)

        def sgs(X):
            _X = torch.tensor(X.astype(np.float32)).reshape(1,8)
            return model(_X).detach().numpy().squeeze()
        
        return cls(sgs=sgs), model.predict(ds_test)

class L96_XY(L96):
    def step(self):
        # fetch Xdot and Ydot from the generator for a given X and Y
        def fn(_X):
            X, Y = _X[:self.K], _X[self.K:]
            Xdot, Xcouple, Ydot = L96_rhs(X, Y, self.F, self.h, self.b, self.c)
            return np.concatenate([Xdot + Xcouple, Ydot])
        
        XY = np.concatenate([self.X, self.Y])
        XY = RK4(fn, self.dt, XY)
        self.X, self.Y = XY[:self.K], XY[self.K:]
        self.t = self.t + self.dt

class ANN(LightningModule):
    def __init__(self, dt=0.005, beta=0, gamma=0, K=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(K,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,K),
            )
        self.dt = dt
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        return self.network(x)
    
    def compute_loss(self, x,y):
        yhat = self(x)
        epsilon = yhat - y
        xhat = x + self.dt * epsilon
        epsilonS = self(xhat) - yhat
        epsilonF = L96_torch(xhat) - L96_torch(x)
        
        MSE = (epsilon * epsilon).mean()
        S_form = (epsilon * epsilonS).mean()
        F_form = (epsilon * epsilonF).mean()
        
        loss = MSE + self.beta * S_form + self.gamma * F_form
        return loss, MSE, S_form, F_form

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss, MSE, S_form, F_form = self.compute_loss(x,y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_MSE', MSE, prog_bar=True)
        self.log('train_S', S_form, prog_bar=True)
        self.log('train_F', F_form, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss, MSE, S_form, F_form = self.compute_loss(x,y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_MSE', MSE, prog_bar=True)
        self.log('val_S', S_form, prog_bar=True)
        self.log('val_F', F_form, prog_bar=True)

    def predict(self, _ds):
        X = torch.tensor(_ds['X'].values.astype('float32'))
        Y = self(X)
        dss = _ds.copy()
        dss['Xpred'] = 0*_ds['Xcouple'] + Y.detach().numpy()
        return dss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    def do_training(self, _train_loader, _val_loader, max_epochs=10, **kw):
        self.trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            **kw
        )
        self.trainer.fit(self, _train_loader, _val_loader)

class ANN_online(LightningModule):
    def __init__(self, dt=0.005, K=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(K,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,K),
            )
        self.dt = dt
        
    def forward(self, x):
        return self.network(x)
    
    def compute_loss(self, x):
        # x has size Nbatch x Nt x K
        # Nt - number of time steps
        func = lambda x: self(x) + L96_torch(x)
        y = 0 * x
        y[:,0:1,:] = x[:,0:1,:]
        for j in range(x.shape[1]-1):
            y[:,j+1:j+2,:] = RK4(func, self.dt, y[:,j:j+1,:])
        
        return ((y-x)**2).mean()

    def training_step(self, batch, batch_nb):
        x, = batch
        loss = self.compute_loss(x)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, = batch
        loss = self.compute_loss(x)
        
        self.log('val_loss', loss, prog_bar=True)

    def predict(self, _ds):
        X = torch.tensor(_ds['X'].values.astype('float32'))
        Y = self(X)
        dss = _ds.copy()
        dss['Xpred'] = 0*_ds['Xcouple'] + Y.detach().numpy()
        return dss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
    def do_training(self, _train_loader, _val_loader, max_epochs=10, **kw):
        self.trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            **kw
        )
        self.trainer.fit(self, _train_loader, _val_loader)