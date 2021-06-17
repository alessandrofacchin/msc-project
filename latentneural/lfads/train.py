from tensorflow.python.types.core import Value
from latentneural.lfads.adaptive_weights import AdaptiveWeights
import time
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .lfads import LFADS


tf.config.run_functions_eagerly(True)

    
def train(model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int, 
    train_dataset: tf.Tensor, adaptive_weights: AdaptiveWeights, 
    val_dataset: Optional[tf.Tensor]=None, batch_size: Optional[int]=None):
    
    assert len(train_dataset) > 0, ValueError('Please provide a non-empty train dataset')
    dims = train_dataset.shape[1:] 
    if val_dataset is not None:
        assert dims == val_dataset.shape[1:], ValueError('Validation and training datasets must have coherent sizes')

    model = LFADS(
        neural_space=dims[-1],
        **model_settings
    )

    model.build(input_shape=[None] + list(dims))

    model.compile(
        optimizer=optimizer,
        loss_weights=adaptive_weights.w
    )

    model.fit(
        x=train_dataset, 
        y=None, 
        callbacks=[adaptive_weights], 
        shuffle=True, 
        epochs=epochs, 
        validation_data=(val_dataset, None),
        batch_size=batch_size
    )

    return model