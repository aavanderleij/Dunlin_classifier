"""

"""

import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class DunlinBehaviorModelTrainer:
    """
    Class for training and validating an image classification model
    """

    def __init__(self):
        self.data_dir = ...  # path to dir with training images
        self.batch_size = 64
        self.img_height = 140
        self.img_width = 180

    def make_train_val_ds(self):
        """
        spits the data set into a training and a validation set
        :return train_ds: tf.data.Dataset object with training data set
        :return val_ds: tf.data.Dataset object with validation data set
        """
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode="rgb")

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            color_mode="rgb")
        return train_ds, val_ds

    def train(self, train_ds, val_ds):
        """
        trains a image classification model
        :param train_ds: tf.data.Dataset object with training data set
        :param val_ds: tf.data.Dataset object with validation data set
        :return:
        """
        logging.debug("training classification model...")

        class_names = train_ds.class_names
        print(class_names)
        for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        logging.debug("autotune")
        AUTOTUNE = tf.data.AUTOTUNE

        logging.debug("casching train_ds")
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        logging.debug("cassing val_ds")
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        logging.debug("rescale layer")
        normalization_layer = layers.Rescaling(1. / 255)

        logging.debug("normalizing")
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        logging.debug("image batch")
        image_batch, labels_batch = next(iter(normalized_ds))

        num_classes = len(class_names)

        logging.debug("setup model")
        model = Sequential([
            # TODO add more preprocessing layers
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            keras.layers.RandomFlip("horizontal"),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        logging.info("compile model")
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        logging.info("display model summary")
        model.summary()

        # amount of epochs (change if needed)
        epochs = 10
        logging.debug(f"amount of epochs = {epochs}")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # get amount of epochs for plotting
        epochs_range = range(epochs)

        # make a plot to show accuracy during training
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        # save plot
        plt.title('Training and Validation Accuracy')

        # make a plot to show loss during training
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # save plot
        plt.savefig('3_classes_epoch_10_v2.png')

        # save model
        model.save('test_model.keras')

        model.save('brigt_model.keras')

        # Save the model.
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

def main():
    # call class
    dbm = DunlinBehaviorModelTrainer()
    # train test split
    train_ds, val_ds = dbm.make_train_val_ds()
    # train model
    dbm.train(train_ds, val_ds)


if __name__ == "__main__":
    sys.exit(main())
