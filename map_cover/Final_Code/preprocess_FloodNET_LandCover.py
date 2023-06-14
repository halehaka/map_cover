import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

IMG_SIZE = 256
CHANNELS = 3
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 15000


def display_sample(display_list, index):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Aug Image', 'Aug Mask']
    # make a color bar
    palette = np.array([[0, 0, 0],  # black - Background
                        [255, 0, 0],  # red - building
                        [0, 255, 0],  # green - woodland
                        [0, 0, 255],  # blue - water
                        [255, 255, 255],  # white - road
                        [255, 255, 0]])  # yellow - car

    fig, axs = plt.subplots(1, len(display_list))
    for i in range(len(display_list)):
        display_list[i] = tf.squeeze(display_list[i])
        axs[i].set_title(title[i] + str(index))
        if i == 0 or i == 2:
            axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        if i == 1 or i == 3:
            axs[i].imshow(palette[np.int16(display_list[i].numpy())])
        axs[i].axis('off')
    plt.show()


@tf.function
def augmentation(input_image, input_mask):
    # random crop and resize
    if tf.random.uniform(()) > 0.5:
        crop_size = np.random.randint(8 * IMG_SIZE // 10, 9 * IMG_SIZE // 10, dtype=int)

        input_image = tf.image.random_crop(input_image, size=(crop_size, crop_size, 3))
        input_image = tf.image.resize(input_image, [IMG_SIZE, IMG_SIZE])

        input_mask = tf.image.resize(input_mask, [IMG_SIZE, IMG_SIZE])
        input_mask = tf.image.random_crop(input_mask, size=(crop_size, crop_size, 1))
        input_mask = tf.image.resize(input_mask, [IMG_SIZE, IMG_SIZE])

    # random hue
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_hue(input_image, 0.3)

    # random saturation
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_saturation(input_image, 5, 15)

    # sharpness (image quality decrease)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_jpeg_quality(input_image, 75, 95)
        input_image = tf.reshape(input_image, (IMG_SIZE, IMG_SIZE, 3))

    # random brightness adjustment illumination
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_brightness(input_image, 0.5)

    # random contrast adjustment
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_contrast(input_image, 0.2, 0.5)

    # random horizontal flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    # random vertical flip
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    # random grayscale
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.rgb_to_grayscale(input_image)
        input_image = tf.image.grayscale_to_rgb(input_image)

    # random noise
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(input_image), mean=0.0, stddev=0.1, dtype=tf.float32)
        input_image = tf.add(input_image, noise)

    #     # rotation in 30° steps
    #     if tf.random.uniform(()) > 0.5:
    #         rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
    #         angle = np.pi/12*rot_factor
    #         input_image = tfa.image.rotate(input_image, angle)
    #         input_mask = tfa.image.rotate(input_mask, angle)

    return input_image, input_mask


def parse_image_FloodNET(train_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    # img_path = os.path.join(BASE_DIR, 'output')
    image = tf.io.read_file(train_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    # print(image)

    # For one Image path:
    # ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # ADE_train_00000001_m.png

    mask_path = tf.strings.regex_replace(train_path, ".jpg", "_lab.png")
    # print(mask_path)
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 1, np.dtype('uint8').type(1), mask) # building flooded to building
    mask = tf.where(mask == 3, np.dtype('uint8').type(4), mask) # road flooded to road
    mask = tf.where(mask == 2, np.dtype('uint8').type(1), mask) # building flooded to building
    mask = tf.where(mask == 5, np.dtype('uint8').type(3), mask) # water to water
    mask = tf.where(mask == 6, np.dtype('uint8').type(2), mask) # tree to woodland
    mask = tf.where(mask == 7, np.dtype('uint8').type(5), mask) # vechile to vechile
    mask = tf.where(mask == 8, np.dtype('uint8').type(3), mask) # pool to water
    mask = tf.where(mask == 9, np.dtype('uint8').type(0), mask) # grass to background
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}


def parse_image_LandCover(train_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    # img_path = os.path.join(BASE_DIR, 'output')
    image = tf.io.read_file(train_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    # print(image)

    # For one Image path:
    # ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # ADE_train_00000001_m.png

    mask_path = tf.strings.regex_replace(train_path, ".jpg", "_m.png")
    # print(mask_path)
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}


def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask



def data_loading(BASE_DIR_FloodNET,BASE_DIR_LandCover, train_ids_LandCover, val_ids_LandCover,train_ids_FloodNET, val_ids_FloodNET, aug=0, show_samples=False):
    # load and parse FloodNET:
    train_paths_FloodNET = [BASE_DIR_FloodNET + '/' + i.rstrip() + '.jpg' for i in train_ids_FloodNET]
    train_dataset_FloodNET = tf.data.Dataset.from_tensor_slices(train_paths_FloodNET)
    train_dataset_FloodNET = train_dataset_FloodNET.map(parse_image_FloodNET)
    train_dataset_FloodNET = train_dataset_FloodNET.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_paths_FloodNET = [BASE_DIR_FloodNET + '/' + i.rstrip() + '.jpg' for i in val_ids_FloodNET]
    val_dataset_FloodNET = tf.data.Dataset.from_tensor_slices(val_paths_FloodNET)
    val_dataset_FloodNET = val_dataset_FloodNET.map(parse_image_FloodNET)
    val_dataset_FloodNET = val_dataset_FloodNET.map(load_image_train)

    # load and parse LandCover
    train_paths_LandCover = [BASE_DIR_LandCover + '/output/' + i.rstrip() + '.jpg' for i in train_ids_LandCover]
    train_dataset_LandCover = tf.data.Dataset.from_tensor_slices(train_paths_LandCover)
    train_dataset_LandCover = train_dataset_LandCover.map(parse_image_LandCover)
    train_dataset_LandCover = train_dataset_LandCover.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_paths_LandCover = [BASE_DIR_LandCover + '/output/' + i.rstrip() + '.jpg' for i in val_ids_LandCover]
    val_dataset_LandCover = tf.data.Dataset.from_tensor_slices(val_paths_LandCover)
    val_dataset_LandCover = val_dataset_LandCover.map(parse_image_LandCover)
    val_dataset_LandCover = val_dataset_LandCover.map(load_image_train)

    # Concatenate:

    train_dataset = train_dataset_LandCover.concatenate(train_dataset_FloodNET)
    val_dataset = val_dataset_LandCover.concatenate(val_dataset_FloodNET)

    dataset = {"train": train_dataset, "val": val_dataset}

    # -- Train Dataset --#
    for i in range(aug):
        aug_dataset = dataset['train'].map(augmentation)
        if show_samples:
            for k, (augm, norm) in enumerate(zip(aug_dataset, dataset['train'])):
                display_list = [norm[0], norm[1], augm[0], augm[1]]
                display_sample(display_list, k)
        dataset['train'] = dataset['train'].concatenate(aug_dataset)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validation Dataset --#
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    return dataset


if __name__ == '__main__':
    data_loading()
