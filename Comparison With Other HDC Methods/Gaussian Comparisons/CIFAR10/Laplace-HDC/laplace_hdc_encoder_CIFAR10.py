import numpy as np
import torch
from torch.utils.data import TensorDataset
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
class hdc_encoder:
    """Memory‐efficient HDC encoder with streaming shift_1d."""
    def __init__(self, inputdim, hyperdim, K, mode):
        self.inputdim  = inputdim
        self.hyperdim  = hyperdim
        self.K         = K
        self.mode      = mode
        self.encode_batch = None
        self.sz        = None
        self.random_permutations = None

        assert mode in ["shift_1d", "shift_2d", "block_diag_shift_1d",
                        "block_diag_shift_2d", "rand_permutation"]

        # pick your streaming implementation for shift_1d:
        if mode == "shift_1d":
            self.encode_batch = self.shift_1d_stream
            self.sz = (1, hyperdim)
        # … keep your other modes the same, or refactor them similarly …
        elif mode == "rand_permutation":
            self.random_permutations = torch.stack(
                [torch.randperm(hyperdim, device=device) for _ in range(inputdim)]
            )
            self.encode_batch = self.random_permutation_stream
            self.sz = (1, hyperdim)
        else:
            # (you may leave shift_2d etc. untouched for now)
            self.encode_batch = getattr(self, mode)
            # set self.sz appropriately…

        # build your V lookup once
        num_colors = self.K.shape[0]
        W = np.sin(np.pi/2 * self.K)
        eigvals, eigvecs = np.linalg.eigh(W)
        U = (np.sqrt(np.maximum(0, eigvals))[:,None] * eigvecs.T)
        U = torch.from_numpy(U).float().to(device)
        G = torch.randn(hyperdim, num_colors, device=device)
        V = torch.sign(G @ U)               # [hyperdim × num_colors]
        self.V = (V >= 0).T.bool().to(device)  # [num_colors × hyperdim]

    def encode(self, loader):
        n = len(loader.dataset)
        Ux     = torch.zeros(n, self.hyperdim, dtype=torch.bool, device=device)
        labels = torch.zeros(n, dtype=torch.long, device=device)
        i0 = 0

        for imgs, labs in loader:
            b = imgs.size(0)
            i1 = i0 + b

            # quantize floats→[0..num_colors-1], flatten:
            if imgs.dtype.is_floating_point:
                batch_data = (imgs.clamp(0,1)*(self.K.shape[0]-1)).round().long()
            else:
                batch_data = imgs.long()
            batch_data = batch_data.view(b, self.inputdim)

            # add one dim so encode_batch sees shape (b, 1, inputdim)
            batch_data = batch_data.unsqueeze(1).to(device)

            # streaming encode
            Ux[i0:i1] = self.encode_batch(batch_data)
            labels[i0:i1] = labs.to(device)
            i0 = i1

        return TensorDataset(Ux.cpu(), labels.cpu())

    def shift_1d_stream(self, x):
        """
        Memory‐efficient 1D shift:
          x: (batch,1,inputdim) of color‐indices
        """
        x = x.squeeze(1)   # → (batch, inputdim)
        batch, d = x.shape
        U = torch.zeros(batch, self.hyperdim, dtype=torch.bool, device=device)

        for i in range(d):
            idx        = x[:, i]             # (batch,)
            Pi         = self.V[idx]         # (batch, hyperdim)
            Pi_shifted = torch.roll(Pi, shifts=i, dims=1)
            U ^= Pi_shifted                  # XOR in‐place

        return U

    def random_permutation_stream(self, x):
        """
        Streaming version of your rand_permutation:
          x: (batch,1,inputdim)
        """
        x = x.squeeze(1)
        batch, d = x.shape
        U = torch.zeros(batch, self.hyperdim, dtype=torch.bool, device=device)

        for i in range(d):
            idx   = x[:, i]                   # (batch,)
            Pi    = self.V[idx]              # (batch, hyperdim)
            Pi_r  = Pi[:, self.random_permutations[i]]
            U ^= Pi_r

        return U

    # ... keep your other methods (shift_2d, block_diag_*, etc.) as-is ...
