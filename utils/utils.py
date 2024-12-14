import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import gdown


cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def set_global_seed(seed_value):
    # Set random seed for Python's random module
    random.seed(seed_value)
    
    # Set random seed for NumPy
    np.random.seed(seed_value)
    
    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)


def random_rotate(image, rotation_range):
    """Rotate the image within the specified range and fill blank space with nearest neighbor."""
    # Generate a random rotation angle in radians
    theta = tf.random.uniform([], -rotation_range, rotation_range) * tf.constant(3.14159265 / 180, dtype=tf.float32)

    # Get the image dimensions
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Create the rotation matrix
    rotation_matrix = tf.stack([
        [tf.cos(theta), -tf.sin(theta), 0],
        [tf.sin(theta),  tf.cos(theta), 0],
        [0, 0, 1]
    ])

    # Adjust for center-based rotation
    translation_to_origin = tf.stack([
        [1, 0, -width / 2],
        [0, 1, -height / 2],
        [0, 0, 1]
    ])
    
    translation_back = tf.stack([
        [1, 0, width / 2],
        [0, 1, height / 2],
        [0, 0, 1]
    ])

    # Cast matrices to tf.float32 for compatibility
    rotation_matrix = tf.cast(rotation_matrix, tf.float32)
    translation_to_origin = tf.cast(translation_to_origin, tf.float32)
    translation_back = tf.cast(translation_back, tf.float32)

    # Perform matrix multiplication
    transform_matrix = tf.linalg.matmul(translation_back, tf.linalg.matmul(rotation_matrix, translation_to_origin))

    # Extract the affine part of the transformation matrix (2x3 matrix for 2D transformation)
    affine_matrix = transform_matrix[:2, :]

    # Flatten the matrix into a 1D array and add [0, 0] to make it 8 elements
    affine_matrix_8 = tf.concat([affine_matrix[0, :], affine_matrix[1, :], [0, 0]], axis=0)

    # Apply the transformation with `fill_mode="nearest"`
    rotated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(affine_matrix_8, [1, 8]),
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST"
    )

    return tf.squeeze(rotated_image)


def random_translate(image, width_factor, height_factor):
    """Randomly translate the image horizontally and vertically within the specified factors.
    
    Args:
        image: Input image tensor.
        width_factor: Horizontal shift factor (0.1 means 10% of width).
        height_factor: Vertical shift factor (0.1 means 10% of height).
    
    Returns:
        Translated image.
    """
    # Get the image dimensions
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]

    # Convert factors to tensors and cast them to float32
    width_factor = tf.cast(width_factor, tf.float32)
    height_factor = tf.cast(height_factor, tf.float32)

    # Cast image dimensions to float32 to match the factor types
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # Calculate the maximum shifts based on the image dimensions
    max_width_shift = width * width_factor
    max_height_shift = height * height_factor

    # Generate random translation values within the given factors
    tx = tf.random.uniform([], -max_width_shift, max_width_shift, dtype=tf.float32)
    ty = tf.random.uniform([], -max_height_shift, max_height_shift, dtype=tf.float32)

    # Create the translation matrix as a 1D array with 8 values
    # [a, b, tx, d, e, ty, 0, 0]
    translation_matrix = tf.concat([
        tf.ones([1], dtype=tf.float32),  # a = 1
        tf.zeros([1], dtype=tf.float32),  # b = 0
        [tx],                             # tx (horizontal shift)
        tf.zeros([1], dtype=tf.float32),  # d = 0
        tf.ones([1], dtype=tf.float32),   # e = 1
        [ty],                             # ty (vertical shift)
        tf.zeros([2], dtype=tf.float32)   # [0, 0]
    ], axis=0)

    # Apply the translation with `fill_mode="nearest"`
    translated_image = tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, 0),
        transforms=tf.reshape(translation_matrix, [1, 8]),  # Ensure 8 values
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="NEAREST"
    )

    return tf.squeeze(translated_image)


def grid_plot(X, y=None, y_preds=None, class_names=None, scaling_factor=2.5, total_items_to_show=25, seed=None):
    random.seed(seed)
    if y is not None and y.shape[1] > 1:
        y = y.argmax(axis=1)

    if y_preds is not None and y_preds.shape[1] > 1:
        y_preds = y_preds.argmax(axis=1)

    if isinstance(X, str):
        directory_path = X
        directory_items = os.listdir(directory_path)
        total_items_to_show = min(len(directory_items), total_items_to_show)
        rand_indices = random.sample(range(len(directory_items)), total_items_to_show)
        X = []
        # for item_name in directory_items[:total_items_to_show]:
        for rand_idx in rand_indices:
            # item_path = f"{directory_path}/{item_name}"
            item_path = f"{directory_path}/{directory_items[rand_idx]}"
            image = cv2.imread(item_path)[..., ::-1]
            X.append(image)                                          
    elif not isinstance(X, (list, np.ndarray)):
        raise(Exception("Either provide an array of images or \
                        a path to a directory containing images as the first argument."))

    # X = X[:total_items_to_show]
    total_items_to_show = min(len(X), total_items_to_show)
    rand_indices = random.sample(range(len(X)), total_items_to_show)
    cols = int(np.ceil(np.sqrt(total_items_to_show)))
    rows = int(np.ceil(total_items_to_show/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(scaling_factor*cols, scaling_factor*rows))
    for i, rand_idx in enumerate(rand_indices):
        ax = axes[i//cols][i%cols]
        image = X[rand_idx]
        ax.imshow(image)
        if y is not None:
            if y_preds is not None:
                if y_preds[rand_idx].item() == y[rand_idx].item():
                    ax.set_title(class_names[y[rand_idx].item()], color="green")
                else:
                    ax.set_title(class_names[y_preds[rand_idx].item()], color="red")
                    ax.text(0, 2, class_names[y[rand_idx].item()], color='white', bbox=dict(facecolor='green'))
            else:
                ax.set_title(class_names[y[rand_idx].item()])
        ax.axis("off")


def download_model(model_url, model_path):
    FILE_ID = model_url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        gdown.download(download_url, model_path)
    print(f"Model Downloaded Successfully to {model_path}!")