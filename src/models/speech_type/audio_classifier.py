import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score

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
        
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        where p_t = p if y=1, else 1-p
    """
    def __init__(self, gamma=3.0, alpha=0.5, name='improved_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        loss = alpha_t * focal_weight * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
    
    
# --- Custom Macro F1 Metric ---
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

    Attributes:
        precisions (list): List of Precision metrics, one per class
        recalls (list): List of Recall metrics, one per class
        
    Returns:
        float: Macro-averaged F1-score across all classes
        
    Formula:
        F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        Macro_F1 = (1/N) * Σ F1_i
    """
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.precisions = [Precision(thresholds=threshold, name=f'precision_{i}') for i in range(num_classes)]
        self.recalls = [Recall(thresholds=threshold, name=f'recall_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            y_true_class = y_true[..., i]
            y_pred_class = y_pred[..., i]
            self.precisions[i].update_state(y_true_class, y_pred_class, sample_weight)
            self.recalls[i].update_state(y_true_class, y_pred_class, sample_weight)

    def result(self):
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precisions[i].result()
            r = self.recalls[i].result()
            f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores) if f1_scores else tf.constant(0.0, dtype=tf.float32)

    def reset_state(self):
        for i in range(self.num_classes):
            self.precisions[i].reset_state()
            self.recalls[i].reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)
    
# --- Threshold Optimizer ---
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    """
    Keras callback for optimizing classification thresholds during training.
    
    Automatically finds optimal decision thresholds for each class to maximize F1-score.
    Runs threshold optimization every 5 epochs on validation data.
    
    Args:
        validation_generator: Validation data generator for threshold optimization
        mlb_classes (list): List of multi-label class names
        
    Attributes:
        best_thresholds (list): Current best thresholds for each class
        best_f1 (float): Best macro F1-score achieved
        
    Note:
        Saves optimized thresholds to 'thresholds.json' in model log directory.
        Uses F1-score as optimization criterion for each class independently.
    """
    def __init__(self, validation_generator, mlb_classes):
        super().__init__()
        self.validation_generator = validation_generator
        self.mlb_classes = mlb_classes
        self.best_thresholds = [0.5] * len(mlb_classes)
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.optimize_thresholds()

    def optimize_thresholds(self):
        predictions = self.model.predict(self.validation_generator, verbose=0)
        true_labels = []
        for i in range(len(self.validation_generator)):
            _, labels = self.validation_generator[i]
            true_labels.extend(labels)
        true_labels = np.array(true_labels)
        
        best_thresholds = []
        for class_idx in range(len(self.mlb_classes)):
            best_threshold = 0.5
            best_class_f1 = 0.0
            for threshold in np.arange(0.1, 0.9, 0.01):  # Finer granularity
                pred_binary = (predictions[:, class_idx] > threshold).astype(int)
                f1 = f1_score(true_labels[:, class_idx], pred_binary, zero_division=0)
                if f1 > best_class_f1:
                    best_class_f1 = f1
                    best_threshold = threshold
            best_thresholds.append(best_threshold)
        
        self.best_thresholds = best_thresholds
        print(f"Optimized thresholds: {dict(zip(self.mlb_classes, best_thresholds))}")
        # Save thresholds
        with open(os.path.join(self.model.log_dir, 'thresholds.json'), 'w') as f:
            json.dump(dict(zip(self.mlb_classes, best_thresholds)), f)