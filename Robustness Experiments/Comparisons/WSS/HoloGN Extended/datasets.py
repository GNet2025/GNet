# Press the green button in the gutter to run the script.
import numpy as np

def load_small_mnist():
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits = load_digits()
    digits_data = digits['data']
    digits_data_rounded = (digits_data > 7.5).astype(np.int8)
    target = digits['target']

    plt.gray()
    plt.matshow(digits_data_rounded[0].reshape(8,8))
    plt.show()

    return (digits_data_rounded,target)

def load_large_mnist():
    from sklearn.datasets import fetch_openml
    import matplotlib.pyplot as plt
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    digits_data_rounded = (X > 128).astype(np.int8)
    target = y.astype(np.int8)

    plt.gray()
    plt.matshow(digits_data_rounded[0].reshape(28,28))
    plt.show()

    return (digits_data_rounded,target)

def load_large_fmnist(mode = 'train'):
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor, Lambda
    import matplotlib.pyplot as plt
    import torch

    # Download and load the FashionMNIST dataset
    trainset = FashionMNIST(root='../../../../../', download=True, train=True, transform=ToTensor())
    # Convert to numpy arrays and threshold to create binary images
    X_train = trainset.data.numpy()
    X_train_rounded = (X_train > 128).astype(int)
    X_train = X_train_rounded.reshape(X_train_rounded.shape[0], -1)
    y_train = trainset.targets.numpy().astype(int)

    # Display the first image
    plt.gray()
    plt.matshow(X_train_rounded[0].reshape(28, 28))
    plt.show()

    testset = FashionMNIST(root='../../../../../', download=True, train=False, transform=ToTensor())
    X_test = testset.data.numpy()
    X_test_rounded = (X_test > 128).astype(int)
    X_test_rounded.reshape(X_test_rounded.shape[0], -1)
    y_test = testset.targets.numpy().astype(int)

    if mode == 'train':
        return (X_train, y_train)
    if mode == 'test':
        return (X_test, y_test)
