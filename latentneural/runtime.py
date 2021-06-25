from __future__ import annotations
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from collections.abc import Iterable
import numpy as np

from latentneural.utils import AdaptiveWeights
from latentneural.models import TNDM, LFADS


tf.config.run_functions_eagerly(True)

class ModelType(Enum):
    TNDM='tndm'
    LFADS='lfads'

    @staticmethod
    def from_string(string: str) -> ModelType:
        if string.lower() == 'tndm':
            return ModelType.TNDM
        elif string.lower() == 'lfads':
            return ModelType.LFADS
        else:
            raise ValueError('Value not recognized.')

    @property
    def with_behaviour(self) -> bool:
        return self.name.lower() == 'tndm'


class Runtime(object):

    @staticmethod
    def clean_datasets(
        train_dataset: Union[List[tf.Tensor], tf.Tensor], 
        val_dataset: Optional[Union[List[tf.Tensor], tf.Tensor]]=None, 
        with_behaviour: bool=False) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Optional[Tuple[tf.Tensor, tf.Tensor]]]:

        train_neural = None
        train_behaviour = None
        valid = None

        if tf.debugging.is_numeric_tensor(train_dataset) or (type(train_dataset) is np.ndarray):
            train_dataset = (train_dataset,)
        elif isinstance(train_dataset, Iterable):
            train_dataset = tuple(train_dataset)
            
        if tf.debugging.is_numeric_tensor(val_dataset) or (type(val_dataset) is np.ndarray):
            val_dataset = (val_dataset,)
        elif isinstance(val_dataset, Iterable):
            val_dataset = tuple(val_dataset)

        if with_behaviour:
            assert len(train_dataset) > 1, ValueError('The train dataset must be a list containing two elements: neural activity and behaviour')
            neural_dims = train_dataset[0].shape[1:]
            behavioural_dims = train_dataset[1].shape[1:]
            train_neural, train_behaviour = train_dataset[:2]
            if val_dataset is not None:
                if len(val_dataset) > 1:
                    assert neural_dims == val_dataset[0].shape[1:], ValueError('Validation and training datasets must have coherent sizes')
                    assert behavioural_dims == val_dataset[1].shape[1:], ValueError('Validation and training datasets must have coherent sizes')
                    valid = val_dataset[:2]
        else:
            assert len(train_dataset) > 0, ValueError('Please provide a non-empty train dataset')
            neural_dims = train_dataset[0].shape[1:] 
            train_neural, train_behaviour = train_dataset[0], None
            if val_dataset is not None:
                assert neural_dims == val_dataset[0].shape[1:], ValueError('Validation and training datasets must have coherent sizes')
                valid = (val_dataset[0], None)
        
        return (train_neural, train_behaviour), valid

    @staticmethod
    def train(model_type: Union[str, ModelType], model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int, 
        train_dataset: Tuple[tf.Tensor, tf.Tensor], adaptive_weights: AdaptiveWeights, 
        val_dataset: Optional[Tuple[tf.Tensor, tf.Tensor]]=None, batch_size: Optional[int]=None, logdir: Optional[str]=None, 
        adaptive_lr: Optional[dict]=None):
        
        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        (x, y), validation_data = Runtime.clean_datasets(train_dataset, val_dataset, model_type.with_behaviour)

        if model_type == ModelType.TNDM:
            model = TNDM(
                neural_space=x.shape[-1],
                behavioural_space=y.shape[-1],
                **model_settings
            )
        elif model_type == ModelType.LFADS:
            model = LFADS(
                neural_space=x.shape[-1],
                **model_settings
            )
        else:
            raise NotImplementedError('This model type has not been implemented yet')

        callbacks = [adaptive_weights]
        if logdir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
        if adaptive_lr is not None:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', **adaptive_lr))

        model.build(input_shape=[None] + list(x.shape[1:]))

        model.compile(
            optimizer=optimizer,
            loss_weights=adaptive_weights.w
        )

        try:
            model.fit(
                x=x,
                y=y,
                callbacks=callbacks,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )
        except KeyboardInterrupt:
            return model

        return model