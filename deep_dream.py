import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DeepDream:
  def __init__(self, layer_names):
    self.model = self._build_model(layer_names)

  def model_summary(self):
    return self.model.summary()

  def calculate_loss(self, image):
    image_batch = tf.expand_dims(image, axis=0)
    layer_activations = self.model(image_batch)

    losses = []
    for activation in layer_activations:
      loss = tf.math.reduce_mean(activation)
      losses.append(loss)

    return tf.reduce_sum(losses)

  @tf.function
  def _deepdream(self, image, step_size):
    with tf.GradientTape() as tape:
      tape.watch(image)
      loss = self.calculate_loss(image)

    gradients = tape.gradient(loss, image)
    gradients /= tf.math.reduce_std(gradients)

    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image

  def run_deep_dream(self, image, steps=2000, step_size=0.001, callback=None):
    image = DeepDream.process_image(image)

    for step in range(steps):
      loss, image = self._deepdream(image, step_size)

      if step % 100 == 0 and callback is not None:
        callback(image, step, loss)

    return DeepDream.deprocess_image(image)

  def run_deep_dream_for_octaves(self, image, steps=200, step_size=0.01,
                                 octave_range=range(-2, 3), octave_scale=1.30, callback=None):
    image = tf.constant(np.array(image))
    base_shape = tf.shape(image)[:-1]
    float_base_shape = tf.cast(base_shape, tf.float32)

    for n in octave_range:
      new_shape = tf.cast(float_base_shape * (octave_scale**n), tf.int32)

      image = tf.image.resize(image, new_shape).numpy()

      image = self.run_deep_dream(image, steps=steps, step_size=step_size, callback=callback)

    return image

  @staticmethod
  def render_iteration(image, step, loss):
    print ("Step {}, loss {}".format(step, loss))
    plt.figure(figsize=(12, 12))
    plt.imshow(DeepDream.deprocess_image(image))
    plt.show()

  @staticmethod
  def process_image(image):
    return tf.keras.applications.inception_v3.preprocess_input(image)

  @staticmethod
  def deprocess_image(image):
    image = 255 * (image + 1.0) / 2.0
    return tf.cast(image, tf.uint8)

  def _build_model(self, layer_names):
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    layers = [base_model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=base_model.input, outputs=layers)
