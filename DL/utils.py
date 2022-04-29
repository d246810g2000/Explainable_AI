"""Helper functions for relevance propagation."""

import tensorflow as tf
import matplotlib.pyplot as plt


def center_crop(image, image_height, image_width):
    """Crops largest central region of image.

    Args:
        image: array of shape (W, H, C)
        image_height: target height of image
        image_width: target width of image

    Returns:
        Cropped image

    Raises:
        Error if image is not of type RGB.
    """

    if (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1):
        raise Exception("Error: Image must be of type RGB.")

    h, w = image.shape[0], image.shape[1]

    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

    return tf.image.resize(cropped_image, (image_width, image_height))


def plot_relevance_map(image, relevance_map, res_dir, i):
    """Plots original image next to corresponding relevance map.

    Args:
        image: original image
        relevance_map: relevance map of original image
        res_dir: path to directory where results are stored
        i: counter
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 2))
    axes[0].imshow(image / 255.0)
    axes[1].imshow(relevance_map, cmap="afmhot")
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    file_path = "{}{}{}".format(res_dir, i, ".png")
    plt.savefig(file_path, dpi=120)
    plt.close(fig)
