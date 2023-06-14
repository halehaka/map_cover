import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from Final_Code.preprocess_FloodNET import data_loading
import random

## Plot Mask Histogram
## Number of Class  4

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

'''
img = cv2.imread('masks/M-33-20-D-d-3-3.tif',cv2.IMREAD_COLOR)
img_arr = np.array(img)  
np.unique(img_arr)
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()

cv2.imshow('masks', img)

cv2.waitKey(0) 
  
#closing all open windows 
cv2.destroyAllWindows() 
'''

def train_FloodNET(train_path, model_name):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        CHANNELS = 3
        BATCH_SIZE = 128
        NUM_CLASSES = 6
        DROPOUT = 0.15
        START_CHANNEL_DIM = 24
        NUM_AUG = 2
        BASE_DIR = train_path

        images_ids = list(map(lambda x: x[:-4], list(filter(lambda x: x[-5] != 'b', os.listdir(BASE_DIR)))))
        masks_ids = list(map(lambda x: x[:-8] ,list(filter(lambda x: x[-5] == 'b', os.listdir(BASE_DIR)))))
        ids = [id for id in images_ids if id in masks_ids]

        random.shuffle(ids)
        val_size = 2 * len(ids)//10
        train_ids, val_ids = ids[val_size:], ids[:val_size]

        dataset = data_loading(train_path, train_ids, val_ids, aug=NUM_AUG)

        # STEPS_PER_EPOCH = len(train_ids) // BATCH_SIZE
        STEPS_PER_EPOCH = len(train_ids * (2**NUM_AUG)) // BATCH_SIZE
        VALIDATION_STEPS = len(val_ids) // BATCH_SIZE

        # Defining U-NET

        inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, CHANNELS))
        s = inputs
        conv1 = tf.keras.layers.Conv2D(START_CHANNEL_DIM, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        conv1 = tf.keras.layers.Dropout(DROPOUT)(conv1)
        conv1 = tf.keras.layers.Conv2D(START_CHANNEL_DIM, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)

        conv2 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
        conv2 = tf.keras.layers.Dropout(DROPOUT)(conv2)
        conv2 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(2, 2)(conv2)

        conv3 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
        conv3 = tf.keras.layers.Dropout(DROPOUT)(conv3)
        conv3 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(2, 2)(conv3)

        conv4 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            pool3)
        conv4 = tf.keras.layers.Dropout(DROPOUT)(conv4)
        conv4 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            conv4)
        pool4 = tf.keras.layers.MaxPooling2D(2, 2)(conv4)

        conv5 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            pool4)
        conv5 = tf.keras.layers.Dropout(DROPOUT)(conv5)
        conv5 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            conv5)

        # EXPANSIVE U-NET

        u6 = tf.keras.layers.Conv2DTranspose(START_CHANNEL_DIM*8, (2, 2), strides=(2, 2), padding='same')(conv5)
        u6 = tf.keras.layers.concatenate([u6, conv4])
        conv6 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        conv6 = tf.keras.layers.Dropout(DROPOUT)(conv6)
        conv6 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(
            conv6)

        u7 = tf.keras.layers.Conv2DTranspose(START_CHANNEL_DIM*4, (2, 2), strides=(2, 2), padding='same')(conv6)
        u7 = tf.keras.layers.concatenate([u7, conv3])
        conv7 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        conv7 = tf.keras.layers.Dropout(DROPOUT)(conv7)
        conv7 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

        u8 = tf.keras.layers.Conv2DTranspose(START_CHANNEL_DIM*2, (2, 2), strides=(2, 2), padding='same')(conv7)
        u8 = tf.keras.layers.concatenate([u8, conv2])
        conv8 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        conv8 = tf.keras.layers.Dropout(DROPOUT)(conv8)
        conv8 = tf.keras.layers.Conv2D(START_CHANNEL_DIM*2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

        u9 = tf.keras.layers.Conv2DTranspose(START_CHANNEL_DIM, (2, 2), strides=(2, 2), padding='same')(conv8)
        u9 = tf.keras.layers.concatenate([u9, conv1])
        conv9 = tf.keras.layers.Conv2D(START_CHANNEL_DIM, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        conv9 = tf.keras.layers.Dropout(DROPOUT)(conv9)
        conv9 = tf.keras.layers.Conv2D(START_CHANNEL_DIM, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
        conv9 = tf.keras.layers.Dropout(DROPOUT)(conv9)
        outputs = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(conv9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        custom_mIoU_metric = CustomMeanIoU(num_classes=NUM_CLASSES, name='mIoU')

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy', custom_mIoU_metric])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}.h5',
                                                        verbose=1,
                                                        save_best_only=True)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')


        results = model.fit(dataset['train'],
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=300, validation_data=dataset['val'],
                            validation_steps=VALIDATION_STEPS,
                            callbacks=[es, checkpoint, tensorboard],
                            )
        print("Starting Retraining with LR = 10e-4")
        model = tf.keras.models.load_model(f'{model_name}.h5', compile=False)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4), loss=tf.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy', custom_mIoU_metric])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}-retrained.h5',
                                                        verbose=1,
                                                        save_best_only=True)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

        results = model.fit(dataset['train'],
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=200, validation_data=dataset['val'],
                            validation_steps=VALIDATION_STEPS,
                            callbacks=[es, checkpoint, tensorboard],
                            )
        print("Starting Retraining with LR = 10e-5")

        model = tf.keras.models.load_model(f'{model_name}-retrained.h5', compile=False)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-5), loss=tf.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy', custom_mIoU_metric])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_name}-retrained-twice.h5',
                                                        verbose=1,
                                                        save_best_only=True)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

        results = model.fit(dataset['train'],
                            steps_per_epoch=STEPS_PER_EPOCH,
                            epochs=200, validation_data=dataset['val'],
                            validation_steps=VALIDATION_STEPS,
                            callbacks=[es, checkpoint, tensorboard],
                         )