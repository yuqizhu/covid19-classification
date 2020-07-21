import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionResNetV2

from utils.image_process import *
import efficientnet.tfkeras as efn

def create_model(model, img_height, img_width, classes):
    if model == 'densenet':
        model = tf.keras.Sequential(
            [DenseNet121(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False),
             tf.keras.layers.GlobalAveragePooling2D(),
             tf.keras.layers.Dense(classes, activation='softmax')]
        )
    elif model == 'efficientnetB7':
        model = tf.keras.Sequential(
            [efn.EfficientNetB7(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False),
             tf.keras.layers.GlobalAveragePooling2D(),
             tf.keras.layers.Dense(classes, activation='softmax')]
        )
    elif model == 'efficientnetB6':
        model = tf.keras.Sequential(
            [efn.EfficientNetB6(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False),
             tf.keras.layers.GlobalAveragePooling2D(),
             tf.keras.layers.Dense(classes, activation='softmax')]
        )

    model.summary()
    weights = model.get_weights()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, weights

def build_lrfn(lr_start=0.000002, lr_max=0.00010,
               lr_min=0, lr_rampup_epochs=5,
               lr_sustain_epochs=15, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * \
                 lr_exp_decay ** (epoch - lr_rampup_epochs \
                                  - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


def train(model, img_width, img_height, classes, epochs, batch_size, train_path, output_path):
    label_path = train_path + '/label.csv'
    train_data = pd.read_csv(label_path)
    rows = train_data.shape[0]

    def process(file_path, label=None):
        file_path = train_path + '/segment/' + file_path
        image_str = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image_str, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        image = tf.image.resize_images(image, (img_height, img_width))
        image = tf.cast(image, tf.float32) / 255.0
        if label is None:
            return image
        return image, label

    train_image = (tf.data.Dataset.from_tensor_slices((train_data['filename'].values, train_data[['healthy', 'covid19']].values))
        .map(process)
        .map(data_augment)
        .repeat()
        .shuffle(512)
        .batch(batch_size))

    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


    STEPS_PER_EPOCH = rows // batch_size
    model, weights = create_model(model, img_height, img_width, classes)
    model.set_weights(weights)
    model.fit(train_image, epochs=epochs, callbacks=[lr_schedule],
              steps_per_epoch=STEPS_PER_EPOCH, verbose=2)
    model.save(output_path)

