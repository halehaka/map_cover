import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from Final_Code.train_FloodNET import CustomMeanIoU

mirrored_strategy = tf.distribute.MirroredStrategy()


def testFloodNET(FloodNET_test_path, model_name, show_samples=True, save_samples=False):
    IMG_SIZE = 256
    CHANNELS = 3
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    NUM_CLASSES = 6

    custom_mIoU_metric = CustomMeanIoU(num_classes=NUM_CLASSES, name='mIoU')
    model = tf.keras.models.load_model(f'{model_name}.h5', compile=False)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy', custom_mIoU_metric])

    test_ids = list(filter(lambda x: x[-5] != 'b', os.listdir(FloodNET_test_path)))

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

    def parse_image(train_path: str) -> dict:
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
        # .../trainset/images/training/ADE_train_00000001.jpg
        # Its corresponding annotation path is:
        # .../trainset/annotations/training/ADE_train_00000001.png

        mask_path = tf.strings.regex_replace(train_path, ".jpg", "_lab.png")
        # print(mask_path)
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)
        # In scene parsing, "not labeled" = 255
        # But it will mess up with our N_CLASS = 150
        # Since 255 means the 255th class
        # Which doesn't exist
        mask = tf.where(mask == 1, np.dtype('uint8').type(1), mask)  # building flooded to building
        mask = tf.where(mask == 3, np.dtype('uint8').type(4), mask)  # road flooded to road
        mask = tf.where(mask == 2, np.dtype('uint8').type(1), mask)  # building flooded to building
        mask = tf.where(mask == 5, np.dtype('uint8').type(3), mask)  # water to water
        mask = tf.where(mask == 6, np.dtype('uint8').type(2), mask)  # tree to woodland
        mask = tf.where(mask == 7, np.dtype('uint8').type(5), mask)  # vechile to vechile
        mask = tf.where(mask == 8, np.dtype('uint8').type(3), mask)  # pool to water
        mask = tf.where(mask == 9, np.dtype('uint8').type(0), mask)  # grass to background

        # Note that we have to convert the new value (0)
        # With the same dtype than the tensor itself

        return {'image': image, 'segmentation_mask': mask}

    def load_image_test(datapoint: dict) -> tuple:
        """Normalize and resize a test image and its annotation.

        Notes
        -----
        Since this is for the test set, we don't need to apply
        any data augmentation technique.

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

    test_paths = [FloodNET_test_path + '/' + i.rstrip() for i in test_ids]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(parse_image)

    dataset = {'test': test_dataset}

    dataset['test'] = dataset['test'].map(load_image_test)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    def display_sample(display_list, index, show_sample, save_sample):
        """Show side-by-side an input image,
        the ground truth and the prediction.
        """
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'True Mask', 'Predicted Mask']
        # make a color map of fixed colors
        tal = colors.ListedColormap(['red', 'yellow', 'green', 'blue', 'black'])
        # make a color bar
        palette = np.array([[0, 0, 0],  # black - Background
                            [255, 0, 0],  # red - building
                            [0, 255, 0],  # green - woodland
                            [0, 0, 255],  # blue - water
                            [255, 255, 255],  # white - road
                            [255, 255, 0]])  # yellow - car
        fig, axs = plt.subplots(1, len(display_list))
        for i in range(len(display_list)):
            axs[i].set_title(title[i] + str(index))
            if i == 0:
                axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap=tal)
            if i == 1:
                axs[i].imshow(palette[np.int16(display_list[i].numpy())])
            if i == 2:
                axs[i].imshow(palette[np.int16(display_list[i])])
            axs[i].axis('off')
        if show_sample:
            plt.show()
        if save_sample:
            plt.savefig(f"/examples/{model_name}_on_FloodNET_{str(index)}.jpg")

    y_pred = model.predict(dataset['test'], steps=len(test_ids) // BATCH_SIZE)
    y_pred = np.argmax(y_pred, axis=3)

    _, acc, meanIoU = model.evaluate(dataset['test'], steps=len(test_ids) // BATCH_SIZE)
    print("Accuracy: ", (acc * 100.0), "%")
    print("Mean IOU: ", (meanIoU * 100.0), "%")

    for i, (image_batch, mask_batch) in enumerate(dataset['test']):
        for j in range(BATCH_SIZE):
            k = j + i * BATCH_SIZE
            sample_image, sample_mask, sample_pred = image_batch[j], mask_batch[j], y_pred[k]
            display_sample([sample_image, tf.squeeze(sample_mask), sample_pred], k, show_samples, save_samples)
