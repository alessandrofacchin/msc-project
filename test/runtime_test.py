import tensorflow as tf
import numpy as np
import pytest
import os

from latentneural.runtime import Runtime, ModelType
from latentneural.utils import AdaptiveWeights


@pytest.mark.unit
def test_train_wrap_tndm():
    Runtime.train(
        model_type=ModelType.TNDM,
        model_settings={}, 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=(np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), np.exp(np.random.randn(100, 100, 4))), 
        val_dataset=(np.random.binomial(1, 0.5, (2, 100, 50)).astype(float), np.exp(np.random.randn(2, 100, 4))), 
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 0],
            update_rate=[0, 0.002, 0.002, 0, 0],
        ),
        batch_size=20
    )

@pytest.mark.unit
def test_train_wrap_tndm_different_specs():
    Runtime.train(
        model_type='tndm',
        model_settings=dict(
            encoded_var_max=0.1,
            original_generator=False
        ), 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=(np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), np.exp(np.random.randn(100, 100, 4))), 
        val_dataset=None,
        adaptive_lr=dict(factor=0.95, patience=10, min_lr=1e-5),
        logdir=os.path.join('.','latentneural','data','storage'),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0, 0, 0],
            update_rate=[0, 0.002, 0.002, 0, 0],
        ),
        batch_size=20
    )


@pytest.mark.unit
def test_train_wrap_lfads():
    Runtime.train(
        model_type='lfads',
        model_settings={}, 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), 
        val_dataset=np.random.binomial(1, 0.5, (20, 100, 50)).astype(float), 
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0],
            update_rate=[0, 0.002, 0],
        ),
        batch_size=20
    )

@pytest.mark.unit
def test_train_wrap_lfads_different_specs():
    Runtime.train(
        model_type=ModelType.LFADS,
        model_settings=dict(
            encoded_var_max=0.1,
            original_generator=False
        ), 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=np.random.binomial(1, 0.5, (100, 100, 50)).astype(float),
        val_dataset=None,
        adaptive_lr=dict(factor=0.95, patience=10, min_lr=1e-5),
        logdir=os.path.join('.','latentneural','data','storage'),
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0],
            update_rate=[0, 0.002, 0.002],
        ),
        batch_size=20
    )