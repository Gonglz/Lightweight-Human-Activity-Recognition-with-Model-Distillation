import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Create output directory
output_dir = "output_offline"
os.makedirs(output_dir, exist_ok=True)

# Log file path
log_file_path = os.path.join(output_dir, "training_log.txt")
log_file = open(log_file_path, "w")

def log_message(message):
    print(message)
    log_file.write(message + "\n")

# Hyperparameters
alpha = 0.5
temperature = 5.0
batch_size = 32
dropout_rate = 0.3
l2_weight = 0.009

# Load data and soft labels
data = np.load('dataset/preprocessed_train_test.npz')
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
teacher_soft_labels = np.load('output/teacher_soft_labels.npy')
log_message(f"Data loaded:\nX_train: {X_train.shape}, y_train: {y_train.shape}\nX_test: {X_test.shape}, y_test: {y_test.shape}")

# Custom distillation loss function
def distillation_loss(y_true, y_pred, teacher_soft_labels):
    y_hard = y_true
    batch_size = tf.shape(y_pred)[0]
    teacher_soft_batch = tf.gather(teacher_soft_labels, tf.range(batch_size))
    hard_loss = tf.keras.losses.CategoricalCrossentropy()(y_hard, y_pred)
    teacher_soft_temp = tf.nn.softmax(teacher_soft_batch / temperature)
    y_pred_temp = tf.nn.softmax(y_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(teacher_soft_temp, y_pred_temp)
    return alpha * hard_loss + (1 - alpha) * soft_loss

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

# Learning rate scheduler
lr_schedule = ExponentialDecay(initial_learning_rate=0.00008, decay_steps=10000, decay_rate=0.9)

# Compile the model
student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, teacher_soft_labels),
    metrics=['accuracy']
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)

# Train the model
history = student_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=30,
    callbacks=[early_stopping]
)

# Save the model
model_path = os.path.join(output_dir, "student_model_regularized.h5")
student_model.save(model_path)
log_message(f"Model saved at: {model_path}")

# Model evaluation
y_pred = student_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Save classification report
classification_report_str = classification_report(y_true_classes, y_pred_classes)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(classification_report_str)
log_message(f"Classification report saved at: {report_path}")

# Plot and save confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()
log_message(f"Confusion matrix saved at: {conf_matrix_path}")

# Plot and save training and validation curves
plt.figure(figsize=(14, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

training_curves_path = os.path.join(output_dir, "training_curves.png")
plt.savefig(training_curves_path)
plt.close()
log_message(f"Training curves saved at: {training_curves_path}")

log_file.close()
print(f"Outputs saved in directory: {output_dir}")
