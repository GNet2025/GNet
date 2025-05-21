import math
from typing import Union

import torch

from . import spatial
from . import Encoder           # ← the updated Encoder that accepts *perturbation*
from . import _fasthd


class OnlineHD(object):
    r"""
    Hyperdimensional classifier (OnlineHD).

    A `(classes, dim)` tensor ``self.model`` stores one *class hypervector*
    per class.  Training updates those vectors; inference computes cosine
    similarity between an encoded sample and every class hypervector.

    Parameters
    ----------
    classes  : int > 0
        Number of distinct classes.
    features : int > 0
        Dimensionality of the original data fed to the encoder.
    dim      : int > 0, default 4000
        Dimensionality of the high-dimensional (HD) space.

    Notes
    -----
    *Training* always operates on clean (un-perturbed) data.
    *Inference* can optionally flip a user-specified fraction of HD
    components – see the ``perturbation`` parameter in ``predict`` et al.
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, classes: int, features: int, dim: int = 4000):
        self.classes = classes
        self.dim = dim
        self.encoder = Encoder(features, dim)   # already initialised RNG
        self.model = torch.zeros(self.classes, self.dim)

    # =================================================================
    # Public inference API ─ all accept *perturbation* (default = 0.0)
    # =================================================================
    def __call__(self,
                 x: torch.Tensor,
                 encoded: bool = False,
                 *,
                 perturbation: float = 0.0) -> torch.Tensor:
        """Return the predicted class index for each sample in *x*."""
        return self.scores(x, encoded=encoded,
                           perturbation=perturbation).argmax(1)

    # convenient alias
    def predict(self,
                x: torch.Tensor,
                encoded: bool = False,
                *,
                perturbation: float = 0.0) -> torch.Tensor:
        return self(x, encoded=encoded, perturbation=perturbation)

    def probabilities(self,
                      x: torch.Tensor,
                      encoded: bool = False,
                      *,
                      perturbation: float = 0.0) -> torch.Tensor:
        """Return class-probabilities for each sample in *x*."""
        return self.scores(x, encoded=encoded,
                           perturbation=perturbation).softmax(1)

    def scores(self,
               x: torch.Tensor,
               encoded: bool = False,
               *,
               perturbation: float = 0.0) -> torch.Tensor:
        """
        Cosine similarity between *x* (encoded if ``encoded=False``)
        and every class hypervector in ``self.model``.
        """
        if encoded and perturbation:
            raise ValueError("Cannot perturb pre-encoded samples.")

        h = x if encoded else self.encode(x, perturbation=perturbation)
        return spatial.cos_cdist(h, self.model)

    # =================================================================
    # Encoding wrapper
    # =================================================================
    def encode(self,
               x: torch.Tensor,
               *,
               perturbation: float = 0.0) -> torch.Tensor:
        """
        Encode *x* with optional sign-flip noise.  Called by inference
        methods; *training* code never passes ``perturbation`` so
        training data remain clean.
        """
        return self.encoder(x, perturbation=perturbation)

    # =================================================================
    # Training – unchanged (always uses clean encodings)
    # =================================================================
    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            encoded: bool = False,
            lr: float = 0.035,
            epochs: int = 120,
            batch_size: Union[int, None, float] = 1024,
            one_pass_fit: bool = True,
            bootstrap: Union[float, str] = 0.01):
        """
        Train the classifier on (*x*, *y*).  All arguments and behaviour
        are identical to the original implementation.
        """
        h = x if encoded else self.encode(x)  # ← no perturbation!
        if one_pass_fit:
            self._one_pass_fit(h, y, lr, bootstrap)
        self._iterative_fit(h, y, lr, epochs, batch_size)
        return self

    # =================================================================
    # Device / dtype transfer
    # =================================================================
    def to(self, *args, **kwargs):
        """Move internal tensors to another device or dtype."""
        self.model = self.model.to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        return self

    # =================================================================
    # Internal helpers (identical to original implementation)
    # =================================================================
    def _one_pass_fit(self, h, y, lr, bootstrap):
        if bootstrap == 'single-per-class':
            idxs = y == torch.arange(self.classes, device=h.device).unsqueeze_(1)
            banned = idxs.byte().argmax(1)
            self.model.add_(h[banned].sum(0), alpha=lr)
        else:
            cut = math.ceil(bootstrap * h.size(0))
            h_, y_ = h[:cut], y[:cut]
            for lbl in range(self.classes):
                self.model[lbl].add_(h_[y_ == lbl].sum(0), alpha=lr)
            banned = torch.arange(cut, device=h.device)

        n = h.size(0)
        todo = torch.ones(n, dtype=torch.bool, device=h.device)
        todo[banned] = False

        _fasthd.onepass(h[todo], y[todo], self.model, lr)

    def _iterative_fit(self, h, y, lr, epochs, batch_size):
        n = h.size(0)
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                h_ = h[i:i + batch_size]
                y_ = y[i:i + batch_size]

                scores = self.scores(h_, encoded=True)
                y_pred = scores.argmax(1)
                wrong = y_ != y_pred

                ar = torch.arange(h_.size(0), device=h_.device)
                alpha_true = (1.0 - scores[ar, y_]).unsqueeze_(1)
                alpha_pred = (scores[ar, y_pred] - 1.0).unsqueeze_(1)

                for lbl in y_.unique():
                    mask_true = wrong & (y_ == lbl)
                    mask_pred = wrong & (y_pred == lbl)
                    self.model[lbl] += lr * (alpha_true[mask_true] * h_[mask_true]).sum(0)
                    self.model[lbl] += lr * (alpha_pred[mask_pred] * h_[mask_pred]).sum(0)
