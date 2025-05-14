from time import time
import torch
import numpy as np
import onlinehd
from scipy.io import savemat, loadmat
from aeon.datasets import load_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset_name = 'FashionMNIST'
def load():
    from torchvision.datasets import FashionMNIST
    from sklearn.preprocessing import Normalizer
    import torch

    train_ds = FashionMNIST(root='../../Data', train=True, download=True)
    test_ds = FashionMNIST(root='../../Data', train=False, download=True)

    x = train_ds.data.reshape(-1, 28*28).float() / 255.0
    y = train_ds.targets.long()

    x_test = test_ds.data.reshape(-1, 28*28).float() / 255.0
    y_test = test_ds.targets.long()

    scaler = Normalizer().fit(x.numpy())
    x = torch.from_numpy(scaler.transform(x.numpy())).float()
    x_test = torch.from_numpy(scaler.transform(x_test.numpy())).float()

    return x, x_test, y, y_test


def main():
    print('Loading...')
    x, x_test, y, y_test = load()
    print(x.shape, x_test.shape)
    classes = y.unique().size(0)
    features = x.size(1)

    try:
        dims = np.mean(loadmat(f'../EHDGNet_{dataset_name}_nHD.mat')[f'EHDGNet_{dataset_name}_nHD'], axis=1, dtype=int) 
    except:
        dims=range(1000, 5000, 500)
    n_dims = len(dims)
    print(n_dims, 'Number of hyperdimensions to check')
    n_repeats = 20
    split_size = x_test.size(0) // n_repeats
    print(split_size, 'Size of Each Split')
    print(x_test.size(0), 'Whole Size of X Test')
    accuracies = np.zeros((n_dims, n_repeats))
    accuracies_test = np.zeros((n_dims, n_repeats))
    times = np.zeros((n_dims, n_repeats))

    for dim_idx, dim in enumerate(dims):
        for repeat in range(n_repeats):
            split_idx = repeat
            model = onlinehd.OnlineHD(classes, features, dim=dim)
            start_idx = split_idx * split_size
            end_idx   = start_idx + split_size
            
            # x_test_sub = x_test
            # y_test_sub = y_test
            indices = list(range(len(x_test)))
            np.random.seed(42)
            np.random.shuffle(indices)  # or random.shuffle(indices)
            start_idx = split_idx * split_size
            end_idx = start_idx + split_size
            split_indices = indices[start_idx:end_idx]
            x_test_sub = x_test[split_indices]
            y_test_sub = y_test[split_indices]
            if torch.cuda.is_available():
                x_device = x.cuda()
                y_device = y.cuda()
                x_test_device = x_test_sub.cuda()
                y_test_device = y_test_sub.cuda()
                model = model.to('cuda')
            else:
                x_device = x
                y_device = y
                x_test_device = x_test_sub
                y_test_device = y_test_sub

            t_start = time()
            model.fit(x_device, y_device, bootstrap=1.0, lr=0.035, epochs=20)
            t_elapsed = time() - t_start

            yhat = model(x_device)
            yhat_test = model(x_test_device)

            acc = (y_device == yhat).float().mean().item()
            acc_test = (y_test_device == yhat_test).float().mean().item()

            accuracies[dim_idx, repeat] = acc
            accuracies_test[dim_idx, repeat] = acc_test
            times[dim_idx, repeat] = t_elapsed

            print(f"[dim={dim}] Repeat {repeat+1}/{n_repeats}: train_acc={acc:.4f}, test_acc={acc_test:.4f}, time={t_elapsed:.2f}s")

        print(f"=== DIM {dim} SUMMARY ===")
        print(f"Avg train acc: {np.mean(accuracies[dim_idx]):.4f}")
        print(f"Avg test acc: {np.mean(accuracies_test[dim_idx]):.4f}")
        print(f"Avg time: {np.mean(times[dim_idx]):.2f}s\n")

    # Print overall mean accuracy per dimension
    print("Final average test accuracy per dimension:")
    print(np.mean(accuracies_test, axis=1))

    # Save matrices
    if np.max(accuracies_test) < 1:
        savemat(f'{dataset_name}_OnlineHD.mat', {f'{dataset_name}_OnlineHD': accuracies_test * 100})
    else:
        savemat(f'{dataset_name}_OnlineHD.mat', {f'{dataset_name}_OnlineHD': accuracies_test}) 
    print("Results saved to onlinehd_results.mat")

if __name__ == '__main__':
    main()
