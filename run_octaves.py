import numpy as np
import tensorflow as tf
from deep_dream import DeepDream

LAYER_NAMES = ["mixed3", "mixed5"]

def callback(i, s, l):
  print(f"Step: {s}")
  print(f"Loss: {l}")

image = tf.keras.preprocessing.image.load_img("image.png", target_size=(225, 375))
image = np.array(image)

original_image = tf.keras.preprocessing.image.load_img("image.png")
original_image = np.array(original_image)

dd = DeepDream(LAYER_NAMES)

image = dd.run_deep_dream_for_octaves(original_image, steps=5,
                                      step_size=0.01, callback=callback,
                                      original_image=original_image,
                                      reinject_details=True)
