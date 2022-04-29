"""Program to test the LRP algorithm implemented in Tensorflow using pre-trained VGG models.
"""

import os
import yaml
import numpy as np

from PIL import Image
from lrp import RelevancePropagation
from utils import center_crop, plot_relevance_map


def layer_wise_relevance_propagation(conf):

    img_dir = conf["paths"]["image_dir"]
    res_dir = conf["paths"]["results_dir"]

    image_height = conf["image"]["height"]
    image_width = conf["image"]["width"]

    lrp = RelevancePropagation(conf)

    image_paths = list()
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        image_paths += [os.path.join(dirpath, file) for file in filenames]

    for i, image_path in enumerate(image_paths):
        print("Processing image {}".format(i+1))
        image = center_crop(np.array(Image.open(image_path)), image_height, image_width)
        relevance_map = lrp.run(image)
        plot_relevance_map(image, relevance_map, res_dir, i)


def main():
    conf = yaml.safe_load(open("config.yml"))
    layer_wise_relevance_propagation(conf)


if __name__ == '__main__':
    main()
