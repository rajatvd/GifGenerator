"""Functions to generate gifs."""
import logging
import torch
from torch import nn
import imageio

from pytorch_utils.sacred_utils import remove_key

from modules import ODEBlock, ConvODEfunc
from utils import export, get_numbered_filename

# %%
GIFFERS = {}
CONFIGS = {}
# %%

# # %%
# device = 'cuda'
# frames = 100
# end_time = 500
# channels_per_colour = 1
# eps = 1e-3
# image_size = (224, 224)
# smooth_colours = False
# signal.alarm(10)


# %%
@export(GIFFERS, CONFIGS)
def neural_ode(device='cuda',
               frames=100,
               fps=60,
               end_time=500,
               channels_per_colour=1,
               eps=1e-5,
               image_size=(224, 224),
               smooth_colours=False,
               save_dir='gifs/neural_ode_gifs',
               _log=logging.getLogger("neural_ode_gif")):
    """Create a gif using a randomly initialized neural ODE.

    Parameters
    ----------
    device : torch.device
        Device to run gif generation on (the default is 'cuda').
    frames : int
        Number of frames of gif to generate (the default is 100).
    fps : float
        Frames per second (the default is 60).
    end_time : float
        End time of neural ode integration (the default is 500).
    channels_per_colour : int
        Number of channels in the state for each colour. So, there will be
        three times this quantity channels in the state (the default is 1).
    eps : float
        Epsilon to add to calculated deviations to prevent division by zero
        (the default is 1e-5).
    image_size : tuple of 2 ints
        Height and width of the images in the gif (the default is (224, 224)).
    smooth_colours : boolean
        Whether to normalize and then apply tanh for generating images, or
        directly apply tanh. The former approach usually leads to smoother
        range of colours in the gif, but might look less rich.
        (the default is False).

    Returns
    -------
    string
        File path of the created gif.

    """
    channels = channels_per_colour * 3

    with torch.no_grad():

        f = ConvODEfunc(channels, act='relu').to(device).eval()
        odenet = ODEBlock(f).to(device).eval()

        inp = torch.randn(1, channels, *image_size).to(device)
        t = torch.linspace(0, end_time, frames).to(device)
        # t = torch.rand(frames).to(device) * end_time
        # t, _ = t.sort()

        odenet(inp, t)

        _log.info(f"NFE = {odenet.nfe}")

        ims = odenet.outputs
        ims = ims.squeeze()
        r, g, b = torch.split(ims, channels_per_colour, dim=1)
        ims = torch.stack([r, g, b], dim=1).mean(dim=2)

        if smooth_colours:
            flat = ims.view(ims.shape[0], -1)
            avg = flat.mean(dim=1, keepdim=True)
            std = flat.abs().mean(dim=1, keepdim=True)
            normed = (flat - avg) / (std + eps)
            normed = normed.view(ims.shape)
            ims = nn.Tanh()(normed)
            ims = (ims + 1) / 2
        else:
            ims = torch.sigmoid(ims)

        _log.info(f"min = {ims.min().item()}, max = {ims.max().item()}")

        ims = ims.cpu().detach().numpy().transpose([0, 2, 3, 1])
        ims = (ims*255).astype('uint8')

        filename = get_numbered_filename(save_dir, "neural_ode_", ".mp4")

        imageio.mimwrite(filename, ims, fps=fps)

        return filename


remove_key(CONFIGS, "_log")

# %%
# neural_ode()
