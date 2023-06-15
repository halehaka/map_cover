import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
import sys
import tempfile
import math

palette = np.array([[0, 0, 0],  # black - Background
                        [255, 0, 0],  # red - building
                        [0, 255, 0],  # green - woodland
                        [0, 0, 255],  # blue - water
                        [255, 255, 255]])  # white - road

mirrored_strategy = tf.distribute.MirroredStrategy()


model_name = "../resource/v7-LandCover-retrained-twice"

TARGET_SIZE = 512
IMG_WIDTH = 256
IMG_HEIGHT = 256
CHANNELS = 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_CLASSES = 5

class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(CustomMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


def load_model(model_name):
    custom_mIoU_metric = CustomMeanIoU(num_classes=NUM_CLASSES, name='mIoU')
    model = tf.keras.models.load_model(f'{model_name}.h5', compile=False)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy', custom_mIoU_metric])
    return model

def split(img, OUTPUT_DIR : str):
    
    ret = []   
    xy = []    
    
    print("Splitting files into ", OUTPUT_DIR)

    k = 0
    for y in range(0, img.shape[0], TARGET_SIZE):
        for x in range(0, img.shape[1], TARGET_SIZE):
            img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]            

            if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                out_img_path = os.path.join(OUTPUT_DIR, "input_map_patch_{}.jpg".format(k))
                cv2.imwrite(out_img_path, img_tile)
                ret.append(out_img_path)
                xy.append((x,y))

            k += 1
    return img, ret, xy

def normalize(input_image: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].


    Returns
    -------
    tuple
        Normalized image 
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image


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

    return {'image': image}


def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image 

    Returns
    -------
    tuple
        A modified image 
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_WIDTH, IMG_HEIGHT))

    input_image = normalize(input_image)

    return input_image

def predict(model, test_paths):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(parse_image)

    dataset = {'test': test_dataset}

    dataset['test'] = dataset['test'].map(load_image_test)
    dataset['test'] = dataset['test'].repeat()
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=AUTOTUNE)

    y_pred = model.predict(dataset['test'], steps=math.ceil(len(test_paths) / BATCH_SIZE))
    y_pred = np.argmax(y_pred, axis=3)

    return y_pred

def assemble_prediction(image, split_files_xy, pred):
    output_prediction = np.zeros([image.shape[0] // 2, image.shape[1] // 2])
    for (start_x, start_y), patch_pred in zip(split_files_xy, pred):
        output_prediction[start_y // 2 : start_y // 2 + min(IMG_WIDTH, patch_pred.shape[0]), start_x // 2 : start_x // 2 + min(IMG_HEIGHT, patch_pred.shape[1])] = patch_pred
    return output_prediction

def visualize_prediction(output_prediction):
    output_image = palette[np.int16(output_prediction)]
    return output_image
    #

def image_to_pixel_cover(model, image):
    with tempfile.TemporaryDirectory() as OUTPUT_DIR:
        image, split_files, split_files_xy = split(image, OUTPUT_DIR)        
        pred = predict(model, split_files)        
        output = assemble_prediction(image, split_files_xy, pred)
        output_image = visualize_prediction(output)
    return output, output_image

def pixel_labels_to_grid(pixel_labels, minx, miny, maxx, maxy, delta):
    print(pixel_labels.shape)
    delta_x_pixels = math.ceil(pixel_labels.shape[1] / ((maxx - minx) / delta))
    delta_y_pixels = math.ceil(pixel_labels.shape[0] / ((maxy - miny) / delta))
    print("delta: ", delta_x_pixels, delta_y_pixels)
    
    grid = {}
    for j,y in enumerate(range(0,pixel_labels.shape[0], delta_y_pixels)):
        for i,x in enumerate(range(0,pixel_labels.shape[1], delta_x_pixels)):            
            next_x = min(x+delta_x_pixels, pixel_labels.shape[1])
            next_y = min(y+delta_y_pixels, pixel_labels.shape[0])
            patch = pixel_labels[y:next_y, x:next_x]
            grid[i,j] = np.histogram(patch, bins=[x-0.5 for x in range(NUM_CLASSES+1)], density=True)[0]
    return grid


def main(input_image_filename):
    model = load_model(model_name)
    image = cv2.imread(input_image_filename)
    pred, output_image = image_to_pixel_cover(model, image)
    cv2.imwrite("kaka.jpeg", output_image)
    grid = pixel_labels_to_grid(pred, 10.5, 10.5, 20.5, 20.5, 3)
    for i,j in grid:
        print(i, j, grid[i,j])
    return pred
    

if __name__ == '__main__':
    input_image_filename = "kaka_input.tif"
    #sys.argv[1]
    main(input_image_filename)
