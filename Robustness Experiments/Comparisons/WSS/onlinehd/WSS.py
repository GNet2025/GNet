from time import time
import torch
import numpy as np
import onlinehd
from scipy.io import savemat, loadmat
from aeon.datasets import load_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset_name = 'WalkingSittingStanding'
def load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y, metadata = load_classification(dataset_name, return_metadata=True, split='train')
    x_test, y_test = load_classification(dataset_name, split='test')
    if x.shape[0] < 200:
        if x_test.shape[0] >= 200:
            train_size = int((x_train.shape[0] + x_test.shape[0]) * 1/4)
            x_, y_ = load_classification(dataset_name)
            x, y = x_[:train_size, :], y_[:train_size]
            x_test, y_test = x_[train_size:, :], y_[train_size:]

    if y.dtype == object or isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_test = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x.reshape(x.shape[0], -1))
    X_test_scaled = scaler.transform(x_test.reshape(x_test.shape[0], -1))
    
    
    X_min = X_train_scaled.min(axis=0)
    X_max = X_train_scaled.max(axis=0)
    
    denom = (X_max - X_min)
    denom[denom == 0] = 1   # avoid division by zero
    
    X_train_norm = (X_train_scaled - X_min) / denom
    X_test_norm  = (X_test_scaled  - X_min) / denom
    
    # Optional: clip to [0,1] just in case
    X_train_norm = np.clip(X_train_norm, 0, 1)
    X_test_norm  = np.clip(X_test_norm, 0, 1)
    
    x = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    x_test = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
    
    x = x.float()
    y = torch.from_numpy(y).long()
    x_test = x_test.float()
    y_test = torch.from_numpy(y_test).long()
    # print(x.shape, x_test.shape, y.shape, y_test.shape
    return x, x_test, y, y_test

def main():
    print('Loading...')
    x, x_test, y, y_test = load()
    print(x.shape, x_test.shape)
    classes = y.unique().size(0)
    features = x.size(1)

    dim = 15_000
    flip_percs = np.arange(0.0, 0.51, 0.05)
    n_flips = len(flip_percs)
    n_repeats = 20
    split_size = x_test.size(0) // n_repeats
    print(split_size, 'Size of Each Split')
    print(x_test.size(0), 'Whole Size of X Test')
    accuracies = np.zeros((n_flips, n_repeats))
    accuracies_test = np.zeros((n_flips, n_repeats))
    times = np.zeros((n_flips, n_repeats))

    for dim_idx, perc in enumerate(flip_percs):
        for repeat in range(n_repeats):
            split_idx = repeat
            model = onlinehd.OnlineHD(classes, features, dim=dim)
            start_idx = split_idx * split_size
            end_idx   = start_idx + split_size
            
            # x_test_sub = x_test
            # y_test_sub = y_test
            indices = list(range(len(x_test)))
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
            yhat_test = model(x_test_device, perturbation=float(perc))

            acc = (y_device == yhat).float().mean().item()
            acc_test = (y_test_device == yhat_test).float().mean().item()

            accuracies[dim_idx, repeat] = acc
            accuracies_test[dim_idx, repeat] = acc_test
            times[dim_idx, repeat] = t_elapsed

            print(f"[Flip Percentage={np.round(perc, 2)}] Repeat {repeat+1}/{n_repeats}: train_acc={acc:.4f}, test_acc={acc_test:.4f}, time={t_elapsed:.2f}s")

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
