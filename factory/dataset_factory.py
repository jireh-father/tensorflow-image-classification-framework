from slim.dataset import download_and_convert_cifar10
from slim.dataset import download_and_convert_flowers
from slim.dataset import download_and_convert_mnist

dataset_list = ["cifar10", "flowers", "mnist"]


def download_and_make_tfrecord(config):
    if config.dataset_name == 'cifar10':
        download_and_convert_cifar10.run(config)
    elif config.dataset_name == 'flowers':
        download_and_convert_flowers.run(config)
    elif config.dataset_name == 'mnist':
        download_and_convert_mnist.run(config)
