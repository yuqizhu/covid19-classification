import pandas as pd
import tensorflow as tf
import os

def classification(input_path, output_path, model_path, batch_size=6):
    model = tf.keras.models.load_model(model_path)
    img_height = 500
    img_width = 500
    def process(file_path, label=None):
        image_str = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image_str, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        image = tf.image.resize_images(image, (img_height, img_width))
        image = tf.cast(image, tf.float32)  / 255.0
        if label is None:
            return image
        return image, label



    input_images = (tf.data.Dataset.list_files(input_path + '/*')
        .map(process)
        .batch(batch_size))

    image_count = len([f for f in os.listdir(input_path)if os.path.isfile(os.path.join(input_path, f))])

    results = model.predict(input_images, verbose=1, steps=image_count // batch_size)
    results = pd.DataFrame(results)
    filelists = os.listdir(input_path)
    results["filename"] = filelists
    results = results[['filename', 0, 1]]
    results = results.rename(columns={0:"healthy", 1:"covid19"})
    results.to_csv(output_path + '/results.csv')