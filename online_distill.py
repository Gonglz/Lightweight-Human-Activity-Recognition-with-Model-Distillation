import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Create output directory
output_dir = "output_online"
os.makedirs(output_dir, exist_ok=True)

# Set log file path
log_file_path = os.path.join(output_dir, "training_log.txt")
log_file = open(log_file_path, "w")

def log_message(message):
    print(message)
    log_file.write(message + "\n")

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        log_message(f"GPU Setup Error: {e}")

# Hyperparameters
alpha = 0.5
temperature = 5.0
batch_size = 32
dropout_rate = 0.3
l2_weight = 0.009
epochs = 10

# Load data
data = np.load('dataset/preprocessed_train_test.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
log_message(f"Data Shapes:\nX_train: {X_train.shape}, y_train: {y_train.shape}\nX_test: {X_test.shape}, y_test: {y_test.shape}")

# Load teacher model
teacher_model = tf.keras.models.load_model('output/activity_recognition_model_sgd.h5')

# Build student model
student_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=l2(l2_weight)),
    BatchNormalization(),
    Dropout(dropout_rate),
    LSTM(32, return_sequences=False, kernel_regularizer=l2(l2_weight)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Dense(32, activation='relu', kernel_regularizer=l2(l2_weight)),
    Dropout(dropout_rate),
    Dense(y_train.shape[1], activation='softmax', kernel_regularizer=l2(l2_weight))
])

# Define distillation loss function
def distillation_loss(y_true, y_pred, teacher_logits):
    hard_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    teacher_soft = tf.nn.softmax(teacher_logits / temperature)
    y_pred_soft = tf.nn.softmax(y_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(teacher_soft, y_pred_soft)
    return alpha * hard_loss + (1 - alpha) * soft_loss

# Optimizer
optimizer = SGD(learning_rate=ExponentialDecay(initial_learning_rate=0.005, decay_steps=10000, decay_rate=0.9), momentum=0.9)

# Custom training loop
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

for epoch in range(epochs):
    log_message(f"\nEpoch {epoch + 1}/{epochs}")
    progress_bar = tqdm(total=len(train_dataset), desc=f"Epoch {epoch + 1}", unit="step")
    epoch_loss = []
    correct_predictions = 0

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            teacher_logits = teacher_model(x_batch, training=False)
            student_logits = student_model(x_batch, training=True)
            loss = distillation_loss(y_batch, student_logits, teacher_logits)

        gradients = tape.gradient(loss, student_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_weights))

        epoch_loss.append(loss)
        correct_predictions += np.sum(np.argmax(student_logits.numpy(), axis=1) == np.argmax(y_batch.numpy(), axis=1))
        progress_bar.update(1)

    progress_bar.close()
    avg_loss = tf.reduce_mean(epoch_loss).numpy()
    train_accuracy = correct_predictions / len(X_train)
    history["loss"].append(avg_loss)
    history["accuracy"].append(train_accuracy)

    # Validation evaluation
    val_loss_list, correct_predictions = [], 0
    for x_batch, y_batch in val_dataset:
        val_logits = student_model(x_batch, training=False)
        loss = tf.keras.losses.CategoricalCrossentropy()(y_batch, val_logits)
        val_loss_list.append(loss)
        correct_predictions += np.sum(np.argmax(val_logits.numpy(), axis=1) == np.argmax(y_batch.numpy(), axis=1))

    val_loss = tf.reduce_mean(val_loss_list).numpy()
    val_accuracy = correct_predictions / len(X_test)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    log_message(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    log_message(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save student model
student_model_path = os.path.join(output_dir, "student_model_online.h5")
student_model.save(student_model_path)
log_message(f"Student model saved to: {student_model_path}")

# Model evaluation
y_pred = student_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Save classification report
report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(classification_report(y_true_classes, y_pred_classes))
log_message(f"Classification report saved to: {report_path}")

# Save confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
log_message(f"Confusion matrix saved to: {conf_matrix_path}")

# Plot and save training and validation curves
plt.figure(figsize=(14, 5))
# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history["loss"], label='Training Loss')
plt.plot(history["val_loss"], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history["accuracy"], label='Training Accuracy')
plt.plot(history["val_accuracy"], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
training_curves_path = os.path.join(output_dir, "training_curves.png")
plt.savefig(training_curves_path)
plt.close()
log_message(f"Training and validation curves saved to: {training_curves_path}")

log_file.close()
print(f"Outputs saved in directory: {output_dir}")
