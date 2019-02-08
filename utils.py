"""Utilities."""
from functools import wraps, partial
from inspect import signature
import os


# %%
def export(dic, confs=None):
    """Export the function by adding an entry into 'dic'.

    The key is the function name.

    confs: dict
        Optional dictionary into which all keyword only arguments are
        put into as a config dict. Useful for sacred configs.
    """
    def export_decorator(f):
        """Add an entry to 'dic' which points to f."""
        if confs is not None:
            confs[f.__name__] = {}
            sig = signature(f)

            for p in sig.parameters.values():
                if p.default != p.empty:
                    confs[f.__name__][p.name] = p.default

        return wraps(f)(dic.setdefault(f.__name__, f))

    return export_decorator


# # %%
# dirname = "gifs/neural_ode_gifs"
# name = "neural_ode_"
# ext = ".gif"


# %%
def filt(x, name, ext):
    """Return true if filename starts with name and has extenstion ext."""
    path, extension = os.path.splitext(x)
    return path[:len(name)] == name and extension == ext


# %%
def get_numbered_filename(dirname, name, ext):
    """Get the next filename with number in the given directory.

    Parameters
    ----------
    dirname : str
        Directory in which to find the next filename.
    name : str
        First part of the filenames after which to search for the number.
    ext : str
        Extension of the files to search in Eg: '.gif'.

    Returns
    -------
    str
        Next filename.

    """
    all_files = os.listdir(dirname)
    files = filter(partial(filt, name=name, ext=ext), all_files)
    ints = [int(filename[len(name):-len(ext)]) for filename in files]
    count = 1 if len(ints) == 0 else max(ints) + 1
    return os.path.join(dirname, f"{name}{count}{ext}")


# %%
get_numbered_filename('gifs\\neural_ode_gifs', "neural_ode_", ".gif")
