from pathlib import Path

import tensorflow as tf


DATA_DIRECTORY = Path("data/classify")
MODEL_DIRECTORY = Path("model")
MODEL_PATH = MODEL_DIRECTORY / "fish_disease.keras"
CLASSES_PATH = MODEL_DIRECTORY / "classes.txt"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10


def choose_device() -> str:
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print(f"Using TensorFlow GPU device: {gpu_devices[0].name}")
        return "/GPU:0"
    print("No TensorFlow GPU device detected; using CPU.")
    return "/CPU:0"


def main() -> None:
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(
            f"{DATA_DIRECTORY} does not exist. Run download_data.py first."
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_dataset.class_names
    print(f"Classes: {class_names}")
    print(f"Training batches: {tf.data.experimental.cardinality(train_dataset)}")
    print(f"Validation batches: {tf.data.experimental.cardinality(validation_dataset)}")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    augmented_inputs = augmentation(inputs)
    preprocessed_inputs = tf.keras.applications.mobilenet_v2.preprocess_input(
        augmented_inputs
    )
    features = base_model(preprocessed_inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    with tf.device(choose_device()):
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
        )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved classes to: {CLASSES_PATH}")


if __name__ == "__main__":
    main()