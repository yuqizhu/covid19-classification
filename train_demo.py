import os
from cls_train import train

if __name__ == '__main__':
    model = 'densenet'
    img_width = 500
    img_height = 500
    classes = 2
    epochs = 1
    batch_size = 1
    train_path = './input/train_data'

    train(model, img_width, img_height, classes, epochs, batch_size, train_path, "./weight/densenet.h5")