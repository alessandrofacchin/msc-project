import tensorflow as tf
import numpy as np
import pytest

from latentneural import LFADS
from latentneural.lfads.adaptive_weights import AdaptiveWeights
from latentneural.lfads.train import train

@pytest.fixture(scope='function')
def enable_eager():
    tf.compat.v1.enable_eager_execution() # this way, line 317 runs
    return None

@pytest.mark.unit
def test_train_model_quick(enable_eager):
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(float) # val_trials X time X neurons
    
    adaptive_weights = AdaptiveWeights(
        initial=[0.5, 1, 1],
        min_weight=[0., 0., 0.],
        max_weight=[1., 1., 1.],
        update_steps=[1, 2, 1],
        update_starts=[2, 1, 1],
        update_rates=[-0.05, -0.1, -0.01]
    )

    model = LFADS(neural_space=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(x=neural_data_train, y=None, callbacks=[adaptive_weights], shuffle=True, epochs=4, validation_data=(neural_data_val, None))

@pytest.mark.regression
@pytest.mark.slow
def test_training_regression(enable_eager):
    neural_data_train = np.random.binomial(1, 0.5, (10, 100, 50)).astype(float) # test_trials X time X neurons
    neural_data_val = np.random.binomial(1, 0.5, (2, 100, 50)).astype(float) # val_trials X time X neurons
    
    adaptive_weights = AdaptiveWeights(
        initial=[1, 0, 0],
        update_rates=[0, 0.002, 0],
    )

    model = LFADS(neural_space=50, max_grad_norm=200)

    model.build(input_shape=[None] + list(neural_data_train.shape[1:]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss_weights=adaptive_weights.w
    )

    model.fit(x=neural_data_train, y=None, callbacks=[adaptive_weights], shuffle=True, epochs=500, validation_data=(neural_data_val, None))

    log_f, _, _ = model.call(neural_data_train, training=False)

    probs = 1/(1 + np.exp(-log_f.numpy()))
    
    assert np.corrcoef(probs.flatten(), neural_data_train.flatten())[0,1] > 0 # Rates are correlated with actual spikes

@pytest.mark.unit
def test_train_wrap(enable_eager):
    train(
        model_settings={}, 
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        epochs=2, 
        train_dataset=np.random.binomial(1, 0.5, (100, 100, 50)).astype(float), 
        val_dataset=np.random.binomial(1, 0.5, (20, 100, 50)).astype(float), 
        adaptive_weights=AdaptiveWeights(
            initial=[1, 0, 0],
            update_rates=[0, 0.002, 0],
        ),
        batch_size=20
    )