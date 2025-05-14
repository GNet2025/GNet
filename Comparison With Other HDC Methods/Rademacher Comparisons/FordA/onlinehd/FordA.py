from time import time
import torch
import numpy as np
import onlinehd
from scipy.io import savemat

def load():
    traind = np.loadtxt('../FordA_TRAIN.txt')
    x = traind[:, 1:]
    y = traind[:, 0]
    y = np.where(y == 1, 1, 0)

    testd = np.loadtxt('../FordA_TEST.txt')
    x_test = testd[:, 1:]
    y_test = testd[:, 0]
    y_test = np.where(y_test == 1, 1, 0)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    return x, x_test, y, y_test

def main():
    print('Loading...')
    x, x_test, y, y_test = load()
    classes = y.unique().size(0)
    features = x.size(1)

    dims = list(range(5000, 21000, 1000))
    n_dims = len(dims)
    n_repeats = 20

    accuracies = np.zeros((n_dims, n_repeats))
    accuracies_test = np.zeros((n_dims, n_repeats))
    times = np.zeros((n_dims, n_repeats))

    for dim_idx, dim in enumerate(dims):
        for repeat in range(n_repeats):
            model = onlinehd.OnlineHD(classes, features, dim=dim)

            if torch.cuda.is_available():
                x_device = x.cuda()
                y_device = y.cuda()
                x_test_device = x_test.cuda()
                y_test_device = y_test.cuda()
                model = model.to('cuda')
            else:
                x_device = x
                y_device = y
                x_test_device = x_test
                y_test_device = y_test

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
        savemat('FordA_OnlineHD.mat', {'FordA_OnlineHD': accuracies_test * 100})
    else:
        savemat('FordA_OnlineHD.mat', {'FordA_OnlineHD': accuracies_test}) 
    print("Results saved to onlinehd_results.mat")

if __name__ == '__main__':
    main()
