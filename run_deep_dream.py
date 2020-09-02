import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

from deep_dream import DeepDream

ap = argparse.ArgumentParser()

ap.add_argument(
  "--image",
  "-i",
  type=str,
  help="Name of the image"
)
ap.add_argument(
  "--output_image",
  "-o",
  type=str,
  default="/output/output.jpg", # for floydhub
  help="Path for output file"
)
ap.add_argument(
  "--layers",
  "-l",
  nargs="+",
  type=str,
  default=["mixed3", "mixed5"],
  help="Inception v3 layers used for optimization"
)
ap.add_argument(
  "--steps",
  "-s",
  type=int,
  default=2000,
  help="Number of iterations for the deep dream algorithm"
)
ap.add_argument(
  "--step_size",
  "-ss",
  type=float,
  default=0.001,
  help="Similar to learning rate"
)
ap.add_argument(
  "--target_size",
  "-t",
  type=int,
  nargs="+",
  default=[225, 375],
  help="Target size of the image"
)

args = vars(ap.parse_args())

IMAGE = args["image"]
OUTPUT_IMAGE = args["output_image"]
LAYERS = args["layers"]
STEPS = args["steps"]
STEP_SIZE = args["step_size"]
TARGET_SIZE = tuple(args["target_size"])

original_image = tf.keras.preprocessing.image.load_img(IMAGE, TARGET_SIZE)
original_image = np.array(original_image)

dd = DeepDream(LAYERS)

image = dd.run_deep_dream(original_image, STEPS,
                          STEP_SIZE, callback=DeepDream.print_iteration)

im = Image.fromarray(image.numpy())
im.save(OUTPUT_IMAGE)
