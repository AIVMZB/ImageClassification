import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from data_generation import train_data_generator, test_data_generator


def plot_history_loss(history: dict):
    plt.plot(history["loss"], "b", label="Training loss")
    plt.plot(history["val_loss"], "r", label="Validation loss")
    plt.show()


def plot_history_acc(history: dict):
    plt.plot(history["accuracy"], "b", label="Training accuracy")
    plt.plot(history["val_accuracy"], "r", label="Validation accuracy")
    plt.show()


def create_inception_model():
    model = InceptionV3(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    for layer in model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(6, activation="softmax")(x)

    model = tf.keras.Model(model.input, output)

    return model


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    return model


def train_model(epochs=40):
    model = create_inception_model()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    train_generator, validation_generator = train_data_generator()

    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    plot_history_loss(history.history)
    plot_history_acc(history.history)

    model.save("trained_model")

    return model


def train_exising_model(epochs=20):
    model = tf.keras.models.load_model("trained_model")

    train_generator, validation_generator = train_data_generator()

    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    plot_history_loss(history.history)
    plot_history_acc(history.history)

    model.save("trained_model")

    return model


def test_model():
    model: tf.keras.models.Model = tf.keras.models.load_model("trained_model")

    test_generator = test_data_generator()

    metrics = model.evaluate(test_generator)

    print(f"| Loss - {metrics[0]} | Accuracy - {metrics[1]} |")


if __name__ == '__main__':
    train_model(10)
    test_model()
