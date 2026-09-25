import tensorflow as tf


print(f"TensorFlow {tf.__version__} is installed.")

# Learn the simple relationship y = 2x + 1.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer="sgd", loss="mean_squared_error")
features = tf.constant([0, 1, 2, 3], dtype=tf.float32)
labels = tf.constant([1, 3, 5, 7], dtype=tf.float32)
model.fit(features, labels, epochs=200, verbose=0)

prediction = model.predict(tf.constant([10], dtype=tf.float32), verbose=0)[0][0]
print(f"Prediction for x=10: {prediction:.2f} (expected about 21.00)")
