import tensorflow as tf
import numpy as np
import pytest

from latentneural import LFADS
from latentneural.lfads.train import train_step, train_model

   
@pytest.mark.unit
def test_train_step():
    neural_data = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # trials X time X neurons
    
    model = LFADS(neural_space=50)
    model.build(input_shape=[None] + list(neural_data.shape[1:]))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_step(model, neural_data, optimizer)

@pytest.mark.unit
def test_train_model_quick():
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(float) # val_trials X time X neurons
    
    model = LFADS(neural_space=50)
    model.build(input_shape=[None, 100, 50])

    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_model(
        model,
        optimizer,
        epochs=1,
        train_dataset=[neural_data_train],
        val_dataset=[neural_data_val],
        coefficients=[5,1,1,1]
    )

@pytest.mark.regression
@pytest.mark.slow
def test_training_regression():
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(float) # val_trials X time X neurons
    
    model = LFADS(neural_space=50, behaviour_space=2)
    model.build(input_shape=[None, 100, 50])

    optimizer = tf.keras.optimizers.Adam(1e-3)

    train_model(
        model,
        optimizer,
        epochs=1000,
        train_dataset=[neural_data_train],
        val_dataset=[neural_data_val],
        coefficients=[5,1,1,1]
    )

    log_f, _ = model.call(neural_data_train, training=False)

    probs = 1/(1 + np.exp(-log_f.numpy()))
    
    assert np.corrcoef(probs.flatten(), neural_data_train.flatten())[0,1] > 0 # Rates are correlated with actual spikes