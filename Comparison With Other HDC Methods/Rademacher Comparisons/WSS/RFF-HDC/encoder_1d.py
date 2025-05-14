import torch
import numpy as np
import scipy.stats
import time

class RandomFourierEncoder:
    def __init__(self, input_dim, gamma, gorder=2, output_dim=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.gorder = gorder
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def pts_map(self, x, r=1.0):
        theta = 2.0 * np.pi / self.gorder * x
        pts = r * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return pts

    def GroupRFF(self, x, sigma):
        intervals = sigma * torch.tensor(
            [scipy.stats.norm.ppf(i / self.gorder) for i in range(1, self.gorder)]
        ).float()
        # print('Thresholds for discretizing Fourier features into group elements:', intervals)
        group_index = torch.zeros_like(x)
        group_index[x <= intervals[0]] = 0
        group_index[x > intervals[-1]] = self.gorder - 1
        if self.gorder > 2:
            for i in range(1, self.gorder - 1):
                group_index[(x > intervals[i - 1]) & (x <= intervals[i])] = i
        return group_index

    def build_item_mem(self):
        correction_factor = 1 / 1.4
        x = np.linspace(0, 255, num=256)
        Cov = np.array([np.exp(-correction_factor * self.gamma ** 2 * ((x - y) / 255.0) ** 2 / 2) for y in range(256)])
        k = Cov.shape[0]
        assert Cov.shape[1] == k, "Covariance matrix must be square."
        L = np.sin(Cov * np.pi / 2.0)
        eigen_values, eigen_vectors = np.linalg.eigh(L)
        R = eigen_vectors @ np.diag(np.maximum(0, eigen_values) ** 0.5) @ eigen_vectors.T
        item_mem = torch.from_numpy(np.random.randn(self.output_dim, k) @ R).float()
        sigma = np.sqrt((R ** 2).sum(0).max())
        self.item_mem = self.GroupRFF(item_mem, sigma).T.to(self.device)
        # print(f"self.item_mem.shape: {self.item_mem.shape}")
        return self.item_mem

    def encode_one_img(self, x):
        '''
        x: [batch_size, input_dim]
        Returns: [batch_size, output_dim]
        '''
        x = x.to(self.device).long()  # indices
        bs, num_features = x.shape  # e.g., [32, 618]
        
        # index into item_mem: shape [batch_size, num_features, output_dim]
        rv = self.item_mem[x].view(bs, num_features, -1)

        # apply circular shifts for each feature
        for i in range(num_features):
            rv[:, i, :] = torch.roll(rv[:, i, :], shifts=num_features - i, dims=-1)

        # sum over features → [batch_size, output_dim]
        rv = torch.sum(rv, dim=1)

        if self.gorder == 2:
            rv = rv % 2

        return rv

    def encode_data_extract_labels(self, dataset):
        '''
        dataset: TensorDataset of (x, y)
        Returns: encoded features [N, output_dim], labels [N]
        '''
        n_samples = len(dataset)
        rv = torch.zeros((n_samples, self.output_dim), device=self.device)
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        # print('Starting encoding data...')
        start_time = time.time()

        batch_size = 128
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, (x_batch, y_batch) in enumerate(loader):
            num_batch = x_batch.size(0)
            rv[i * batch_size: i * batch_size + num_batch] = self.encode_one_img((x_batch * 255).int())
            labels[i * batch_size: i * batch_size + num_batch] = y_batch
            # if (i + 1) % 10 == 0 or i == 0:
            #     print(f"Encoded {i * batch_size + num_batch}/{n_samples} samples (elapsed {time.time() - start_time:.2f}s)")
        
        # print('Encoding complete.')
        return rv, labels

    def group_bind(self, lst):
        return torch.sum(lst, dim=0)

    def group_bundle(self, lst):
        intervals = torch.tensor([2 * np.pi / self.gorder * i for i in range(self.gorder)]) + np.pi / self.gorder
        pts = torch.sum(self.pts_map(lst), dim=0)
        raw_angles = 2 * np.pi + torch.arctan(pts[:, 1] / pts[:, 0]) - np.pi * (pts[:, 0] < 0).float()
        angles = torch.fmod(raw_angles, 2 * np.pi)
        return torch.floor(angles / (2.0 * np.pi) * self.gorder + 0.5)

    def similarity(self, x, y):
        return torch.sum(torch.sum(self.pts_map(x) * self.pts_map(y), dim=-1), dim=-1) * (1.0 / x.size(-1))