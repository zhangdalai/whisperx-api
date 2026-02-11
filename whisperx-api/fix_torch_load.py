"""
Monkey patch torch.load to use weights_only=False by default.
This is needed for compatibility with older model checkpoints that use
Python objects like omegaconf.ListConfig.
"""
import torch
import functools

# Store original torch.load
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that forces weights_only=False for compatibility."""
    # Force weights_only=False regardless of what was passed
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load
print("Applied torch.load patch: weights_only forced to False")
