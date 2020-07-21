import tensorflow as tf

def rotate(image):
    return tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def brightness(image):
    return tf.image.random_brightness(image, 0.1)

def contrast(image):
    return tf.image.random_contrast(image, 0.8, 1.2)

def saturation(image):
    return tf.image.random_saturation(image, 0.8, 1.2)

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = rotate(image)
    if label is None:
        return image
    return image, label