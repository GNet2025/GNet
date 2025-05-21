"""
encoder.py ─ High-dimensional nonlinear encoder with optional
in-place sign-flip perturbation and automatic handling of
multivariate/3-D inputs.

Author: 2025-05-20
"""

import math
import torch


# ──────────────────────────────────────────────────────────────────────
# Utility: in-place sign flipping
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def flip_sign_(tensor: torch.Tensor, percentage: float) -> torch.Tensor:
    """
    Randomly change the sign of `percentage` fraction of entries in-place.

    Parameters
    ----------
    tensor : torch.Tensor
        Any shape; will be modified in-place.
    percentage : float in [0,1]
        Fraction of elements to flip.

    Returns
    -------
    torch.Tensor
        The same tensor (for chaining).
    """
    if percentage <= 0.0:
        return tensor
    if percentage >= 1.0:
        tensor.mul_(-1)
        return tensor

    mask = torch.rand_like(tensor, dtype=torch.float) < percentage
    tensor[mask] *= -1
    return tensor


# ──────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────
class Encoder(object):
    r"""
    Non-linear random projection:
        H_i = cos(X·B_i + b_i) · sin(X·B_i)

    * `B`  ~ 𝒩(0,1)    shape (dim, features)
    * `b`  ~ 𝒰(0,2π)   length dim

    Parameters
    ----------
    features : int
        Dimensionality of the (flattened) input samples.
    dim : int, default 4000
        Dimensionality of the HD space.

    Notes
    -----
    • If the input tensor has more than 2 dimensions (e.g. a multivariate
      time-series of shape (n, channels, length)), the encoder flattens
      each sample on the fly.
    • Use `perturbation` to apply sign-flip noise *after* encoding.
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, features: int, dim: int = 4000):
        if features <= 0 or dim <= 0:
            raise ValueError("features and dim must be positive.")
        self.dim      = dim
        self.features = features
        self.basis    = torch.randn(dim, features)             # B
        self.base     = torch.empty(dim).uniform_(0.0, 2 * math.pi)  # b

    # ──────────────────────────────────────────────────────────────────
    def __call__(self,
                 x: torch.Tensor,
                 *,
                 perturbation: float = 0.0) -> torch.Tensor:
        """
        Encode `x` (shape (n, features) or (n, …)) to shape (n, dim).

        Parameters
        ----------
        x : torch.Tensor
            Input batch.
        perturbation : float in [0,1], default 0.0
            Fraction of HD dimensions to sign-flip per sample.

        Returns
        -------
        torch.Tensor
            Encoded (and optionally perturbed) hyper-vectors.
        """
        if x.dim() > 2:                       # e.g. (n, c, L) → (n, c*L)
            x = x.flatten(start_dim=1)

        if x.size(1) != self.features:
            raise ValueError(
                f"Expected {self.features} features, got {x.size(1)}."
            )
        if not (0.0 <= perturbation <= 1.0):
            raise ValueError("perturbation must be between 0 and 1 inclusive")

        n_samples = x.size(0)
        batch_sz  = max(1, math.ceil(0.01 * n_samples))  # ≈1 % of dataset
        h         = torch.empty(n_samples, self.dim,
                                device=x.device, dtype=x.dtype)
        tmp       = torch.empty(batch_sz,  self.dim,
                                device=x.device, dtype=x.dtype)

        for start in range(0, n_samples, batch_sz):
            chunk = min(batch_sz, n_samples - start)     # exact rows this pass

            # 1) linear projection
            torch.matmul(x[start:start+chunk], self.basis.T, out=tmp[:chunk])

            # 2) add random phase
            torch.add(tmp[:chunk], self.base, out=h[start:start+chunk])

            # 3) non-linear mapping
            h[start:start+chunk].cos_().mul_(tmp[:chunk].sin_())

            # 4) optional perturbation
            flip_sign_(h[start:start+chunk], perturbation)

        return h

    # ──────────────────────────────────────────────────────────────────
    def to(self, *args, **kwargs):
        """
        Move internal tensors to a new device or dtype.

        Examples
        --------
        >>> enc = Encoder(256, 8000).to('cuda', dtype=torch.float16)
        """
        self.basis = self.basis.to(*args, **kwargs)
        self.base  = self.base.to(*args, **kwargs)
        return self
