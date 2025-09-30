import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.metrics import f1_score
from tensorflow.keras.mixed_precision import Policy

# --- Deep Learning Model Architecture ---
def build_model_multi_label(n_mels, fixed_time_steps, num_classes, freeze_cnn=False):
    """
    Build a deep learning model for multi-label voice type classification.
    
    Includes granular control over mixed precision policy to ensure compatibility
    with frozen layers (Conv2D and BatchNormalization).
    
    Parameters:
    ----------
    n_mels (int): 
        Number of mel-frequency bins in input spectrograms (will be capped at 128)
    fixed_time_steps (int): 
        Fixed number of time steps in input spectrograms
    num_classes (int): 
        Number of voice type classes to predict
    freeze_cnn (bool): 
        If True, freeze CNN layers during training to use as fixed feature extractor
        
    Returns
    -------
    Model
        Compiled Keras model ready for training
    """
    effective_n_mels = min(n_mels, 128)
    input_shape = (effective_n_mels + 13, fixed_time_steps, 1)
    
    input_mel = Input(shape=input_shape, name='mel_spectrogram_input')
    x = input_mel

    # Define the mixed precision policy for the currently trainable layers
    # Layers outside the conv_block (RNN/Dense) will inherit the compute_dtype
    # 'mixed_float16' policy sets compute_dtype=float16 and variable_dtype=float32
    mixed_precision_policy = Policy('mixed_float16')

    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1), trainable=True):
        """ResNet-style convolutional block with a trainable flag and explicit policy."""
        shortcut = x
        # For frozen layers, force dtype float32; otherwise, use default (inherits global policy)
        conv_kwargs = {'trainable': trainable}
        bn_kwargs = {'trainable': trainable}
        if not trainable:
            conv_kwargs['dtype'] = 'float32'
            bn_kwargs['dtype'] = 'float32'
        # First convolution + batch norm + activation
        x = Conv2D(filters, kernel_size, strides=strides, padding='same', **conv_kwargs)(x)
        x = BatchNormalization(**bn_kwargs)(x)
        x = Activation('relu', dtype='float32')(x)
        # Second convolution + batch norm
        x = Conv2D(filters, kernel_size, padding='same', **conv_kwargs)(x)
        x = BatchNormalization(**bn_kwargs)(x)
        # Adjust shortcut dimensions if needed
        if shortcut.shape[-1] != filters or strides != (1, 1):
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same', **conv_kwargs)(shortcut)
            shortcut = BatchNormalization(**bn_kwargs)(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu', dtype='float32')(x)
        return x
    
    # Apply the conditional CNN blocks
    x = conv_block(x, 32, trainable=not freeze_cnn)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = conv_block(x, 64, trainable=not freeze_cnn)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = conv_block(x, 128, trainable=not freeze_cnn)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = conv_block(x, 256, trainable=not freeze_cnn)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # --- RNN and Dense Head (Will default to mixed_float16 policy) ---
    
    # Calculate dimensions after CNN layers for reshaping
    # Assuming standard architecture dimensions
    reduced_n_mels = (effective_n_mels + 13) // 16
    reduced_time_steps = fixed_time_steps // 16
    channels_after_cnn = 256

    # Prepare for RNN: reshape (freq, time, channels) -> (time, freq * channels)
    x = Permute((2, 1, 3))(x) 
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    # RNN layers for temporal modeling of audio sequences
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3))(x) 
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)  

    def multi_head_attention(x, num_heads=4):
        """Multi-head attention mechanism to focus on important time steps."""
        head_size = x.shape[-1] // num_heads
        heads = []
        
        for i in range(num_heads):
            attention = Dense(1, activation='tanh')(x)     
            attention = Flatten()(attention)               
            attention = Activation('softmax')(attention)   
            attention = RepeatVector(x.shape[-1])(attention)  
            attention = Permute((2, 1))(attention)         
            
            head = multiply([x, attention])                
            head = Lambda(lambda xin: tf.reduce_sum(xin, axis=1), output_shape=(256,))(head) 
            heads.append(head)
        
        return Concatenate()(heads) if len(heads) > 1 else heads[0]

    x = multi_head_attention(x)

    dense_input = x
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    if dense_input.shape[-1] == 256:
        x = Add()([x, dense_input])
    
    class_outputs = []
    for i in range(num_classes):
        class_branch = Dense(128, activation='relu', name=f'class_{i}_dense')(x)
        class_branch = Dropout(0.3)(class_branch)
        # Final output layer
        class_output = Dense(1, activation='sigmoid', name=f'class_{i}_output')(class_branch)
        class_outputs.append(class_output)
    
    output = Concatenate(name='combined_output')(class_outputs)
    # --- FIX: Ensure final output is float32 for stable logits ---
    output = Activation('sigmoid', dtype='float32')(output) 

    model = Model(inputs=input_mel, outputs=output)
    
    # Recompile the model to apply the new trainable settings
    macro_f1 = MacroF1Score(num_classes=num_classes, name='macro_f1')
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=FocalLoss(gamma=4.0, alpha=0.25),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), macro_f1]
    )
    
    return model

class FocalLoss(tf.keras.losses.Loss):
    """
    Custom Focal Loss implementation for handling class imbalance in multi-label classification.
    
    Focal loss addresses class imbalance by down-weighting easy examples and focusing
    training on hard negatives. Particularly effective for scenarios with significant
    class imbalance or when easier examples dominate the loss.
    
    Parameters
    ---------- 
    gamma (float): 
        Focusing parameter. Higher values down-weight easy examples more
    alpha (float): 
        Weighting factor for rare class. Values in [0,1] for class 1, 1-alpha for class 0
    name (str): 
        Name for the loss function
    """
    def __init__(self, gamma=3.0, alpha=0.5, name='improved_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Convert inputs to float32 for consistent computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0) which would cause NaN
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Compute standard binary cross-entropy loss
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate probability of correct classification (p_t in focal loss formula)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Apply alpha weighting (addresses class imbalance)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Apply focal weight (1-p_t)^gamma (focuses on hard examples)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        
        # Combine all components: FL(p_t) = -α_t * (1-p_t)^γ * CE
        loss = alpha_t * focal_weight * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
    
class MacroF1Score(tf.keras.metrics.Metric):
    """
    Custom Keras metric for computing macro-averaged F1-score in multi-label classification.
    
    Calculates F1-score for each class independently using precision and recall,
    then averages across all classes. Particularly useful for multi-label scenarios
    where equal weight should be given to each class regardless of frequency.
    
    Parameters:
    ----------
    num_classes (int): 
        Number of classes in the multi-label problem
    threshold (float): 
        Decision threshold for binary predictions (default: 0.5)
    name (str): 
        Name for the metric (default: 'macro_f1')

    Returns
    -------
    float: 
        Macro-averaged F1-score across all classes
    """
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        # Ensure metric variables are always float32 for mixed precision compatibility
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to float and apply threshold for binary predictions
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        
        # Compute confusion matrix components for all classes simultaneously
        # True Positives: both true and predicted are 1
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        # False Positives: true is 0 but predicted is 1  
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        # False Negatives: true is 1 but predicted is 0
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        
        # Accumulate counts across batches
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Compute precision and recall for all classes using confusion matrix
        # Precision = TP / (TP + FP) - "Of all positive predictions, how many were correct?"
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        # Recall = TP / (TP + FN) - "Of all actual positives, how many did we find?"
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        
        # F1-score = 2 * (precision * recall) / (precision + recall) - harmonic mean
        f1_scores = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        
        # Return macro average: equal weight to each class regardless of frequency
        return tf.reduce_mean(f1_scores)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)
    
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    """
    Keras callback for optimizing classification thresholds during training.
    
    Automatically finds optimal decision thresholds for each class to maximize F1-score.
    Runs threshold optimization every 5 epochs on validation data.

    Parameters
    ----------
    validation_generator: 
        Validation data generator for threshold optimization
    mlb_classes (list): 
        List of multi-label class names

    Attributes
    ----------
    best_thresholds (list): 
        Current best thresholds for each class
    best_f1 (float): 
        Best macro F1-score achieved
    """
    def __init__(self, validation_generator, mlb_classes):
        super().__init__()
        self.validation_generator = validation_generator
        self.mlb_classes = mlb_classes
        self.best_thresholds = [0.5] * len(mlb_classes)
        self.best_f1 = 0.0
        # Pre-compute threshold candidates for efficiency
        self.threshold_candidates = np.arange(0.1, 0.91, 0.05)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.optimize_thresholds()

    def optimize_thresholds(self):
        """Optimized threshold search using vectorized operations."""
        # Get all predictions and true labels at once to avoid repeated model calls
        predictions = self.model.predict(self.validation_generator, verbose=0)
        true_labels = []
        for batch in self.validation_generator:
            _, labels = batch
            true_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)
        true_labels = np.array(true_labels)
        
        # Handle potential mismatch in sample counts (edge case)
        if len(predictions) != len(true_labels):
            min_samples = min(len(predictions), len(true_labels))
            predictions = predictions[:min_samples]
            true_labels = true_labels[:min_samples]
        
        best_thresholds = []
        
        # Find optimal threshold for each class independently
        for class_idx in range(len(self.mlb_classes)):
            y_true_class = true_labels[:, class_idx]
            y_pred_class = predictions[:, class_idx]
            
            # Skip optimization if no positive samples exist (F1 undefined)
            if np.sum(y_true_class) == 0:
                best_thresholds.append(0.5)  # Use default threshold
                continue
            
            # Test each threshold candidate and compute F1-score
            f1_scores = []
            for threshold in self.threshold_candidates:
                pred_binary = (y_pred_class > threshold).astype(int)
                f1 = f1_score(y_true_class, pred_binary, zero_division=0)
                f1_scores.append(f1)
            
            # Select threshold that maximizes F1-score
            best_idx = np.argmax(f1_scores)
            best_threshold = self.threshold_candidates[best_idx]
            best_thresholds.append(float(best_threshold))
        
        self.best_thresholds = best_thresholds
        
        # Compute overall macro F1 with optimized thresholds
        macro_f1 = self._compute_macro_f1(predictions, true_labels, best_thresholds)
        
        print(f"Optimized thresholds (Macro F1: {macro_f1:.4f}): {dict(zip(self.mlb_classes, best_thresholds))}")
        
        # Save thresholds and performance for later use
        threshold_dict = dict(zip(self.mlb_classes, best_thresholds))
        threshold_dict['macro_f1'] = float(macro_f1)
        
        if self.model.log_dir is not None and Path(self.model.log_dir).exists():
            with open(Path(self.model.log_dir) / 'thresholds.json', 'w') as f:
                json.dump(threshold_dict, f, indent=2)

    def _compute_macro_f1(self, predictions, true_labels, thresholds):
        """Compute macro F1-score with given thresholds.
        
        Macro F1: Average of per-class F1 scores (treats all classes equally).
        This is different from micro F1 which would pool all predictions together.
        """
        f1_scores = []
        for class_idx, threshold in enumerate(thresholds):
            y_true = true_labels[:, class_idx]
            y_pred = (predictions[:, class_idx] > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        return np.mean(f1_scores)  # Equal weight to each class
    
    def reset_state(self):
        self.best_thresholds = [0.5] * len(self.mlb_classes)
        self.best_f1 = 0.0

    def get_config(self):
        config = super().get_config()
        config.update({'mlb_classes': self.mlb_classes})
        return config

    @classmethod
    def from_config(cls, config):
        mlb_classes = config.pop('mlb_classes', [])
        return cls(validation_generator=None, mlb_classes=mlb_classes)