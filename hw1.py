import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# ======================
# 1. Load the dataset
# ======================
dataset, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
all_data = dataset["train"]   # 25k images

# ======================
# 2. Split into Train/Val/Test
# ======================
# 80% train, 10% validation, 10% test
train_size = int(0.8 * info.splits["train"].num_examples)
val_size   = int(0.1 * info.splits["train"].num_examples)
test_size  = int(0.1 * info.splits["train"].num_examples)

train_ds = all_data.take(train_size)
rest     = all_data.skip(train_size)
val_ds   = rest.take(val_size)
test_ds  = rest.skip(val_size)

# ======================
# 3. Preprocessing
# ======================
IMG_SIZE = (160, 160)

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    return image, label

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (train_ds
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .shuffle(1000)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

test_ds = (test_ds
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

# ======================
# 4. Build the Model
# ======================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160,160,3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze pretrained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ======================
# 5. Train the Model
# ======================
history = model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds
)

# ======================
# 6. Evaluate on Test Set
# ======================
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test Accuracy: {test_acc:.3f}")
