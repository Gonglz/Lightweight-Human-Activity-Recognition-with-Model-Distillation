import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Output directory
output_dir = 'output/'
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 64
epochs = 30
dropout_rate = 0.2
l2_weight = 0.003
learning_rate = 0.00008

# Load data
data_path = 'dataset/preprocessed_train_test.npz'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

data = np.load(data_path)
X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

# Build the model
teacher_model = Sequential([
    GaussianNoise(0.1, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_weight)),
    BatchNormalization(),
    Dropout(dropout_rate),
    LSTM(64, return_sequences=False, kernel_regularizer=l2(l2_weight)),
    BatchNormalization(),
    Dropout(dropout_rate),
    Dense(64, activation='relu', kernel_regularizer=l2(l2_weight)),
    Dropout(dropout_rate),
    Dense(y_train.shape[1], activation='softmax', kernel_regularizer=l2(l2_weight))
])

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

# Compile the model
teacher_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Set early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=9,
    restore_best_weights=True
)

# Train the model
history = teacher_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Save the model
teacher_model_path = os.path.join(output_dir, 'activity_recognition_model_sgd.h5')
teacher_model.save(teacher_model_path)
print(f"Teacher model saved to {teacher_model_path}")

# Save soft labels
teacher_soft_labels = teacher_model.predict(X_train)
soft_labels_path = os.path.join(output_dir, 'teacher_soft_labels.npy')
np.save(soft_labels_path, teacher_soft_labels)
print(f"Soft labels saved to {soft_labels_path}")

# Evaluate the model and save results to a file
evaluation_output_path = os.path.join(output_dir, 'evaluation_output.txt')
with open(evaluation_output_path, 'w') as f:
    y_pred = teacher_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Classification report
    classification_rep = classification_report(y_true_classes, y_pred_classes)
    print("\nClassification Report:")
    print(classification_rep)
    f.write("Classification Report:\n")
    f.write(classification_rep + "\n\n")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion Matrix saved to {confusion_matrix_path}")
    f.write(f"Confusion Matrix saved to {confusion_matrix_path}\n")

# Save training and validation curves
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

training_validation_curves_path = os.path.join(output_dir, 'training_validation_curves.png')
plt.savefig(training_validation_curves_path)
print(f"Training and validation curves saved to {training_validation_curves_path}")

# Save path information to file
with open(evaluation_output_path, 'a') as f:
    f.write(f"\nTraining and validation curves saved to {training_validation_curves_path}\n")
