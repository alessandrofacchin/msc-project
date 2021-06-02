import tensorflow as tf
import numpy as np

from latentneural import TNDM
from latentneural.tndm.train import train_step, train_model

   
def test_train_step():
    neural_data = np.exp(np.random.randn(10, 100, 50)) # trials X time X neurons
    behaviour_data = np.exp(np.random.randn(10, 100, 2)) # trials X time X behaviour
    
    model = TNDM(neural_space=50, behaviour_space=2)
    model.build(input_shape=[None] + list(neural_data.shape[1:]))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_step(model, neural_data, behaviour_data, optimizer)

def test_training():
    neural_data_train = np.exp(np.random.randn(100, 100, 50)) # test_trials X time X neurons
    behaviour_data_train = np.exp(np.random.randn(100, 100, 2)) # test_trials X time X behaviour
    neural_data_val = np.exp(np.random.randn(20, 100, 50)) # val_trials X time X neurons
    behaviour_data_val = np.exp(np.random.randn(20, 100, 2)) # val_trials X time X behaviour
    
    model = TNDM(neural_space=50, behaviour_space=2)
    model.build(input_shape=[None, 100, 50])

    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_model(
        model, 
        optimizer, 
        epochs=100, 
        train_dataset=[(neural_data_train, behaviour_data_train)],
        val_dataset=[(neural_data_val, behaviour_data_val)]
    )