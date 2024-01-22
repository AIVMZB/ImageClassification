from keras.preprocessing.image import ImageDataGenerator
from config import DATASET_PATH
import os

TRAIN_DIR = os.path.join(DATASET_PATH, "seg_train")
TEST_DIR = os.path.join(DATASET_PATH, "..", "seg_test", "seg_test")


def train_data_generator():
    image_generator = ImageDataGenerator(
        rescale=1. / 255.,
        rotation_range=45,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    train_generator = image_generator.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(128, 128),
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        subset="training"
    )

    validation_generator = image_generator.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(128, 128),
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        subset="validation"
    )

    return train_generator, validation_generator


def test_data_generator():
    image_data_generator = ImageDataGenerator(
        rescale=1. / 255.
    )

    data_generator = image_data_generator.flow_from_directory(
        directory=TEST_DIR,
        target_size=(128, 128),
        class_mode="categorical",
        batch_size=32
    )

    return data_generator
