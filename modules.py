"""Neural ODE pytorch modules."""

import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

# %%
ACTS = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    }


# %%
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=True)


def norm(dim):
    """Group norm with minimum group size of 32."""
    return nn.GroupNorm(min(32, dim), dim)


# %%
class ODEfunc(nn.Module):
    """Basic 4 layer ODE function with group norm.

    dim is the number of channels in ode state."""

    def __init__(self, dim, act='relu'):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.norm2 = norm(dim)
        self.conv2 = conv3x3(dim, dim)
        self.norm3 = norm(dim)
        self.conv3 = conv3x3(dim, dim)
        self.norm4 = norm(dim)
        self.conv4 = conv3x3(dim, dim)
        self.norm5 = norm(dim)
        self.nfe = torch.tensor(0)

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm4(out)
        out = self.relu(out)
        out = self.conv4(out)

        out = self.norm5(out)
        return out


# %%
class ConvODEfunc(nn.Module):
    """Two convolution network for an ODE function.
    Inputs and outputs are the same size.

    Parameters
    ----------
    dim : int
        Number of channels in input (and output).
    act : string
        Activation function. One of relu, sigmoid or tanh
        (the default is 'relu').
    """

    def __init__(self, dim, act='relu'):
        super().__init__()
        self.act = ACTS[act]()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(dim)
        self.nfe = torch.tensor(0)

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(x)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv3(x)
        out = self.norm3(out)
        out = self.act(out)
        out = self.conv4(out)
        out = self.norm4(out)
        return out
        # return (out - out.mean()) / out.std()


# %%
class ODEBlock(nn.Module):
    """Wraps an odefunc into a single module.

    The odefunc must have nfe (number of function evaluations) attribute.

    Parameters
    ----------
    odefunc : nn.Module
        An nn.Module which has an nfe attribute and has same input and output
        sizes.

    rtol: float
        Relative tolerance for ODE evaluations. Default 1e-3

    atol: float
        Absolute tolerance for ODE evaluations. Default 1e-3

    Forward takes x and t as inputs, and returns the final state
    at the given input time points t. Default t is [0, 1]
    self.outputs contains all the outputs at each time step.

    """

    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.t = torch.tensor([0, 1]).float()
        self.outputs = None
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t=None):
        if t is None:
            times = self.t
        else:
            times = t
        self.outputs = odeint(self.odefunc,
                              x,
                              times,
                              rtol=self.rtol,
                              atol=self.atol)
        return self.outputs[1]

    @property
    def nfe(self):
        """Number of function evaluations"""
        return self.odefunc.nfe.item()

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe.fill_(value)
