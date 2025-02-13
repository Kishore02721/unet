import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from dual_unet import unet_with_two_encoders

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9096)]  # 4GB limit
        )
    except RuntimeError as e:
        print(e)

# Define image size and batch size
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
EPOCHS = 50

# Dataset Paths
DATASET_PATH = "dataset"
ORIGINAL_DIR = os.path.join(DATASET_PATH, "original")
INSPECT_DIR = os.path.join(DATASET_PATH, "inspect")
MASK_DIR = os.path.join(DATASET_PATH, "mask")

def load_image(filepath, target_size=IMAGE_SIZE):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, target_size)  # Resize
    img = img.astype("float32") / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def load_dataset(original_dir, inspect_dir, mask_dir):
    original_files = {os.path.splitext(f)[0]: f for f in os.listdir(original_dir)}
    inspect_files = {os.path.splitext(f)[0]: f for f in os.listdir(inspect_dir)}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

    common_filenames = sorted(set(original_files.keys()) & set(inspect_files.keys()) & set(mask_files.keys()))

    images_original, images_inspect, masks = [], [], []

    for name in common_filenames:
        orig_path = os.path.join(original_dir, original_files[name])
        insp_path = os.path.join(inspect_dir, inspect_files[name])
        mask_path = os.path.join(mask_dir, mask_files[name])

        images_original.append(load_image(orig_path))
        images_inspect.append(load_image(insp_path))
        masks.append(load_image(mask_path))

    return np.array(images_original), np.array(images_inspect), np.array(masks)

# Function to compute Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Function to compute Dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Combined BCE + Dice Loss for better performance
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# Function to plot and save the loss graph
def plot_loss(history, save_path="loss_vs_epoch.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Load dataset
X_original, X_inspect, Y_masks = load_dataset(ORIGINAL_DIR, INSPECT_DIR, MASK_DIR)

# Split into train and validation sets (80% train, 20% validation)
X_train_original, X_val_original, X_train_inspect, X_val_inspect, Y_train_masks, Y_val_masks = train_test_split(
    X_original, X_inspect, Y_masks, test_size=0.2, random_state=42
)

# Create a TensorFlow data pipeline
def build_dataset(X_original, X_inspect, Y_masks, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(((X_original, X_inspect), Y_masks))
    dataset = dataset.shuffle(buffer_size=len(X_original))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

class LossLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

# Create train and validation datasets
train_dataset = build_dataset(X_train_original, X_train_inspect, Y_train_masks)
val_dataset = build_dataset(X_val_original, X_val_inspect, Y_val_masks)

# Import the U-Net model
model = unet_with_two_encoders(input_shape=(256, 256, 1), num_classes=1)

# Compile the model with BCE + Dice loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=bce_dice_loss,
              metrics=["accuracy", dice_coefficient])

# Define checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/epoch_{epoch:02d}.h5",
    save_weights_only=True,
    save_freq='epoch'
)

# Train the model with checkpointing
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, LossLoggingCallback()])

plot_loss(history, save_path="loss_vs_epoch.png")

# Save final model
model.save("dual_encoder_unet.h5")   

