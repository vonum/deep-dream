import numpy as np
import tensorflow as tf
from deep_dream import DeepDream

LAYER_NAMES = ["mixed3", "mixed5"]

def callback(i, s, l):
  print(f"Step: {s}")
  print(f"Loss: {l}")

original_image = tf.keras.preprocessing.image.load_img("image.png", target_size=(225, 375))
original_image = np.array(original_image)

dd = DeepDream(LAYER_NAMES)

image = dd.run_deep_dream(original_image, 500, 0.001, callback=callback)
