# used in load_dataset
import cv2
import numpy as np

import tensorflow_datasets as tfds

from filters import darken, gausblur, gausnoise
from corruptor import image_corruptor, corrupt_image_dataset
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader


def display_images(x,y,x_darken,x_blur,x_noise):

    
    # https://cs.stanford.edu/~acoates/stl10/
    # class labels
    for i in range(2):
        labels = {0: "airplane",1: "bird",2: "car", 3: "cat", 4: "deer", 5: "dog", 6: "horse", 
        7: "monkey", 8: "ship", 9: "truck"}
        img = random.randint(0, len(x) -1)
        label = labels[y[img]]


        plt.figure(figsize=(12,4))

        plt.subplot(1,4,1)
        plt.imshow(x[img])
        plt.title(f"Original: {label}")
        plt.axis("off")

        plt.subplot(1,4,2)
        plt.imshow(x_darken[img])
        plt.title(f"Darkened: {label}")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.imshow(x_blur[img])
        plt.title(f"Blurred: {label}")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.imshow(x_noise[img])
        plt.title(f"Noise: {label}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def load_train(train_ds):
    x_train = []
    y_train = []

    for image, label in tfds.as_numpy(train_ds):
        x_train.append(image)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(f"x train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")

    return x_train, y_train

def load_test(test_ds):
    x_test = []
    y_test = []
    for image, label in tfds.as_numpy(test_ds):
        x_test.append(image)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print(f"x test shape: {x_test.shape}")
    print(f"y test shape: {y_test.shape}")

    return x_test, y_test





def load_dataset():
    train_ds, test_ds = tfds.load(
    name = "stl10",
    split = ['train','test'],
    as_supervised=True)

    x_train, y_train = load_train(train_ds)
    x_test, y_test = load_test(test_ds)

    # create corrupted train data calling the corrupt_image_dataset() function
    # from corruptor.py
    x_train_darken_ds = corrupt_image_dataset(x_train,"dark", severity=5)
    x_train_blur_ds = corrupt_image_dataset(x_train,"blur", (3,3))
    x_train_noise_ds = corrupt_image_dataset(x_train, "noise", 15)

    # create corrupted test data 
    x_test_darken_ds = corrupt_image_dataset(x_test,"dark", severity=5)
    x_test_blur_ds = corrupt_image_dataset(x_test,"blur", (3,3))
    x_test_noise_ds = corrupt_image_dataset(x_test, "noise", 15)

    """Normalize
    float point math for gradient compuation in Neural Network"""
    # https://stackoverflow.com/questions/52102692/what-dtype-should-i-use-for-pytorch-parameters-in-a-neural-network-that-inputs-a
    # train set
    x_train = x_train.astype("float32") / 255.0
    x_train_darken_ds = x_train_darken_ds.astype("float32") / 255.0
    x_train_blur_ds = x_train_blur_ds.astype("float32") / 255.0
    x_train_noise_ds = x_train_noise_ds.astype("float32") / 255.0

    # test set
    x_test = x_test.astype("float32") / 255.0
    x_test_darken_ds = x_test_darken_ds.astype("float32") / 255.0
    x_test_blur_ds = x_test_blur_ds.astype("float32") / 255.0
    x_test_noise_ds = x_test_noise_ds.astype("float32") / 255.0

    print("Training dataset")
    display_images(
    x_train,
    y_train,
    x_train_darken_ds,
    x_train_blur_ds,
    x_train_noise_ds
    )

    print("Test dataset")
    display_images(
    x_test,
    y_test,
    x_test_darken_ds,
    x_test_blur_ds,
    x_test_noise_ds
    )



    # https://docs.pytorch.org/docs/2.11/generated/torch.from_numpy.html
    # convert our numpy tensor to pytorch objects
    # train data
    x_train_tensor = torch.from_numpy(x_train)
    x_train_blur_tensor = torch.from_numpy(x_train_blur_ds)
    x_train_noise_tensor = torch.from_numpy(x_train_noise_ds)
    x_train_darken_tensor = torch.from_numpy(x_train_darken_ds)
    y_train_tensor = torch.from_numpy(y_train).long()

    # test data
    x_test_tensor = torch.from_numpy(x_test)
    x_test_blur_tensor = torch.from_numpy(x_test_blur_ds)
    x_test_noise_tensor = torch.from_numpy(x_test_noise_ds)
    x_test_darken_tensor = torch.from_numpy(x_test_darken_ds)
    y_test_tensor = torch.from_numpy(y_test).long()





    """The input size of nn in pytorch (n inputs, channels, Height, Width) 
    and output (n inputs, channels, Height, Width)

    The numpy arrays come in the form (n input, Height, Width, channels)
    so we have to rearrange from
    numpy idx: (0,1,2,3)
    to pytorch idx: (0,3,1,2)
    """
    print("Numpy order: (n inputs, Height, Width, channels)")
    print(f"before permute: {x_train_tensor.shape}\n")
    x_train_tensor = torch.permute(x_train_tensor, ((0,3,1,2)))
    x_train_blur_tensor = torch.permute(x_train_blur_tensor, ((0,3,1,2)))
    x_train_noise_tensor = torch.permute(x_train_noise_tensor, ((0,3,1,2)))
    x_train_darken_tensor = torch.permute(x_train_darken_tensor, ((0,3,1,2)))

    print("pytorch order: (n inputs, channels, Height, Width)")
    print(f"x train after permute: {x_train_tensor.shape}")
    print(f"x train blur after permute: {x_train_blur_tensor.shape}")
    print(f"x train noise after permute: {x_train_noise_tensor.shape}")
    print(f"x train darken after permute: {x_train_darken_tensor.shape}\n")


    # test set permute
    x_test_tensor = torch.permute(x_test_tensor, ((0,3,1,2)))
    x_test_blur_tensor = torch.permute(x_test_blur_tensor, ((0,3,1,2)))
    x_test_noise_tensor = torch.permute(x_test_noise_tensor, ((0,3,1,2)))
    x_test_darken_tensor = torch.permute(x_test_darken_tensor, ((0,3,1,2)))

    print("pytorch order: (n inputs, channels, Height, Width)")
    print(f"x test after permute: {x_test_tensor.shape}")
    print(f"x test blur after permute: {x_test_blur_tensor.shape}")
    print(f"x test noise after permute: {x_test_noise_tensor.shape}")
    print(f"x test darken after permute: {x_test_darken_tensor.shape}")




    return (
        x_train_tensor,
        x_train_blur_tensor,
        x_train_noise_tensor,
        x_train_darken_tensor,
        y_train_tensor,
        x_test_tensor,
        x_test_blur_tensor,
        x_test_noise_tensor,
        x_test_darken_tensor,
        y_test_tensor
    )

