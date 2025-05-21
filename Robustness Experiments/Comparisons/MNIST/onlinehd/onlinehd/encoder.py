import math
import torch

# ----------------------------------------------------------------------
# Helper ────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
@torch.no_grad()
def flip_sign_(tensor: torch.Tensor, percentage: float) -> torch.Tensor:
    """
    In-place sign flip of a random subset of elements.

    Args
    ----
    tensor      : any shape, modified in-place
    percentage  : 0‒1 fraction of elements to flip

    Returns
    -------
    tensor      : the *same* tensor object (for chaining)
    """
    if percentage <= 0.0:
        return tensor
    if percentage >= 1.0:
        tensor.mul_(-1)
        return tensor

    mask = torch.rand_like(tensor, dtype=torch.float) < percentage
    tensor[mask] *= -1
    return tensor


# ----------------------------------------------------------------------
# Updated Encoder ──────────────────────────────────────────────────────
# ----------------------------------------------------------------------
class Encoder(object):
    r"""
    A nonlinear random projection encoder that maps each input vector
    :math:`x\in\mathbb{R}^f` to a high-dimensional hypervector
    :math:`H\in\mathbb{R}^D`:

    .. math::
        H_i = \cos(x\cdot B_i + b_i)\;\sin(x\cdot B_i),

    where the basis matrix :math:`B\sim\mathcal{N}(0,1)` has shape
    ``(dim, features)`` and the phase vector
    :math:`b\sim\mathcal{U}(0,2\pi)` has length ``dim``.

    Parameters
    ----------
    features : int
        Dimensionality of original data (``f``) – must be > 0.
    dim : int, default 4000
        Output dimensionality (``D``) – must be > 0.
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, features: int, dim: int = 4000):
        if features <= 0 or dim <= 0:
            raise ValueError("features and dim must both be positive.")
        self.dim      = dim
        self.features = features
        self.basis    = torch.randn(dim, features)                  # B
        self.base     = torch.empty(dim).uniform_(0.0, 2 * math.pi) # b

    # ──────────────────────────────────────────────────────────────────
    def __call__(
        self,
        x: torch.Tensor,
        *,
        perturbation: float = 0.0,
    ) -> torch.Tensor:
        """
        Encode ``x`` and (optionally) flip a fraction of components.

        Args
        ----
        x            : input tensor of shape ``(n?, features)``.
        perturbation : float in [0,1] – fraction of components whose sign
                       should be flipped per *output* hypervector.

        Returns
        -------
        torch.Tensor : encoded representation of shape ``(n?, dim)``.
        """
        if not (0.0 <= perturbation <= 1.0):
            raise ValueError("perturbation must be between 0 and 1 inclusive")

        n_samples = x.size(0)
        batch_sz  = max(1, math.ceil(0.01 * n_samples))   # ~1 % rule
        h         = torch.empty(n_samples, self.dim,
                                device=x.device, dtype=x.dtype)
        tmp       = torch.empty(batch_sz, self.dim,
                                device=x.device, dtype=x.dtype)

        # use caller-supplied RNG deterministically, but keep global RNG intact
        saved_state = None

        for start in range(0, n_samples, batch_sz):
            end = start + batch_sz
            torch.matmul(x[start:end], self.basis.T, out=tmp[: end - start])
            torch.add(tmp[: end - start], self.base, out=h[start:end])
            h[start:end].cos_().mul_(tmp[: end - start].sin_())

            # ■ apply perturbation
            flip_sign_(h[start:end], perturbation)

        if saved_state is not None:
            torch.random.set_rng_state(saved_state)

        return h

    # ──────────────────────────────────────────────────────────────────
    def to(self, *args, **kwargs):
        """
        Move internal tensors to a new device / dtype (in-place), then
        return *self* for chaining – mirrors the ``torch.nn.Module.to`` API.
        """
        self.basis = self.basis.to(*args, **kwargs)
        self.base  = self.base.to(*args, **kwargs)
        return self
