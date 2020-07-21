imprt pandas as pd
import tensorflow as tf

def classification(input_path, output_path, model_path):
    model = tf.keras.models.load_model(model_path)
    results = model.predict(input_path, batch_size=6, verbose=1)
    results.to_csv()