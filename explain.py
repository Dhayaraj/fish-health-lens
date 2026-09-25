import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("model/fish_disease.keras")
CLASSES_PATH = Path("model/classes.txt")
TEST_DIRECTORY = Path("data/test")
IMAGE_SIZE = (224, 224)


def find_base_model(model: tf.keras.Model) -> tf.keras.Model:
    nested_models = [
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    ]
    for base_model in reversed(nested_models):
        for layer in reversed(base_model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                return base_model
    raise ValueError("Could not find the MobileNetV2 base model.")


def compute_gradcam_plus_plus(
    model: tf.keras.Model,
    image_batch: tf.Tensor,
    class_index: int,
) -> np.ndarray:
    base_model = find_base_model(model)
    last_conv_layer = next(
        layer
        for layer in reversed(base_model.layers)
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
    )
    feature_model = tf.keras.models.Model(
        base_model.input,
        [last_conv_layer.output, base_model.output],
    )
    augmentation = model.get_layer("augmentation")
    base_model_index = model.layers.index(base_model)
    classifier_layers = model.layers[base_model_index + 1 :]

    with tf.GradientTape(persistent=True) as tape:
        augmented_batch = augmentation(image_batch, training=False)
        preprocessed_batch = tf.keras.applications.mobilenet_v2.preprocess_input(
            augmented_batch
        )
        conv_outputs, features = feature_model(preprocessed_batch, training=False)
        predictions = features
        for layer in classifier_layers:
            predictions = layer(predictions, training=False)
        class_score = predictions[:, class_index]
        first_gradients = tape.gradient(class_score, conv_outputs)
        second_gradients = tape.gradient(first_gradients, conv_outputs)
        third_gradients = tape.gradient(second_gradients, conv_outputs)

    del tape
    conv_outputs = conv_outputs[0]
    first_gradients = first_gradients[0]
    second_gradients = second_gradients[0]
    third_gradients = third_gradients[0]

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1), keepdims=True)
    alpha_denominator = 2.0 * second_gradients + third_gradients * global_sum
    alpha_denominator = tf.where(
        tf.abs(alpha_denominator) < 1e-7,
        tf.ones_like(alpha_denominator),
        alpha_denominator,
    )
    alphas = second_gradients / alpha_denominator
    positive_gradients = tf.nn.relu(first_gradients)
    weights = tf.reduce_sum(alphas * positive_gradients, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.maximum(tf.reduce_max(heatmap), 1e-7)
    return heatmap.numpy()


def overlay_heatmap(image_path: Path, heatmap: np.ndarray, output_path: Path) -> None:
    original = Image.open(image_path).convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(original.size)
    heatmap_array = np.asarray(heatmap_image)
    color_map = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
    color_map[..., 0] = heatmap_array
    color_map[..., 1] = np.uint8(255 - heatmap_array)
    overlay = Image.blend(original, Image.fromarray(color_map), alpha=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM++ explanation.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Test image to explain. Defaults to the first image in data/test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("explanations/gradcam_plus_plus.png"),
        help="Path for the heatmap overlay PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image or next(TEST_DIRECTORY.rglob("*.jpg"), None)
    if image_path is None or not image_path.exists():
        raise FileNotFoundError("Provide an image with --image or populate data/test.")
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Train the model first with train.py.")

    class_names = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = tf.expand_dims(
        tf.keras.applications.mobilenet_v2.preprocess_input(image_array), axis=0
    )
    probabilities = model.predict(image_batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    heatmap = compute_gradcam_plus_plus(model, image_batch, class_index)
    overlay_heatmap(image_path, heatmap, args.output)

    print(f"Image: {image_path}")
    print(f"Prediction: {class_names[class_index]}")
    print(f"Confidence: {probabilities[class_index]:.4f}")
    print(f"Saved explanation to: {args.output}")


if __name__ == "__main__":
    main()