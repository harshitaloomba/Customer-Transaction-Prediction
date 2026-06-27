# model.py — Neural Network architecture
# 3-hidden-layer MLP with BatchNorm + Dropout. Compiled for binary classification.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import config


def build_model(input_shape: int) -> Sequential:
    """
    Build and compile a fully-connected neural network.

    Architecture
    ------------
    Dense(256) → BN → ReLU → Dropout(0.4)
    Dense(128) → BN → ReLU → Dropout(0.3)
    Dense(64)  → BN → ReLU → Dropout(0.2)
    Dense(1)   → Sigmoid

    Parameters
    ----------
    input_shape : int   number of input features

    Returns
    -------
    model : compiled Keras Sequential model
    """
    model = Sequential([
        Dense(256, input_dim=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')],
    )
    return model
