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
ap.add_argument(
  "--octave_range",
  "-or",
  type=int,
  nargs="+",
  default=[-2, 3],
  help="How many octaves to use"
)
ap.add_argument(
  "--octave_scale",
  "-os",
  type=float,
  default=1.30,
  help="Octave increase ratio"
)
ap.add_argument(
  "--reinject_details",
  "-r",
  action="store_true",
  help="Whether to reinject details after each deep dream cycle"
)

args = vars(ap.parse_args())

IMAGE = args["image"]
OUTPUT_IMAGE = args["output_image"]
LAYERS = args["layers"]
STEPS = args["steps"]
STEP_SIZE = args["step_size"]
TARGET_SIZE = tuple(args["target_size"])
OCTAVE_RANGE = range(args["octave_range"][0], args["octave_range"][1])
OCTAVE_SCALE = args["octave_scale"]
REINJECT_DETAILS = args["reinject_details"]

original_image = tf.keras.preprocessing.image.load_img(IMAGE)
original_image = np.array(original_image)

image = tf.keras.preprocessing.image.load_img(IMAGE, target_size=TARGET_SIZE)
image = np.array(image)

dd = DeepDream(LAYERS)

image = dd.run_deep_dream_for_octaves(
  image,
  steps=STEPS,
  step_size=STEP_SIZE,
  octave_range=OCTAVE_RANGE,
  octave_scale=OCTAVE_SCALE,
  callback=DeepDream.print_iteration,
  original_image=original_image,
  reinject_details=REINJECT_DETAILS
)

im = Image.fromarray(image.numpy())
im.save(OUTPUT_IMAGE)
