from __future__ import annotations

from collections import defaultdict
from latentneural.utils.args_parser import ArgsParser
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from collections.abc import Iterable
import numpy as np
import json
import yaml
import os
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import Ridge
from datetime import datetime
import getpass
import socket

from latentneural import TNDM, LFADS
from latentneural.utils import AdaptiveWeights, logger, CustomEncoder
from latentneural.data import DataManager
import latentneural.losses as lnl


tf.config.run_functions_eagerly(True)


class ModelType(Enum):
    TNDM = 'tndm'
    LFADS = 'lfads'

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

    @property
    def weights_num(self) -> bool:
        return 5 if self.name.lower() == 'tndm' else 3


class Runtime(object):

    @staticmethod
    def clean_datasets(
            train_dataset: Union[List[tf.Tensor], tf.Tensor],
            val_dataset: Optional[Union[List[tf.Tensor], tf.Tensor]] = None,
            with_behaviour: bool = False) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Optional[Tuple[tf.Tensor, tf.Tensor]]]:

        train_neural = None
        train_behaviour = None
        valid = None

        if tf.debugging.is_numeric_tensor(train_dataset) or (
                isinstance(train_dataset, np.ndarray)):
            train_dataset = (train_dataset,)
        elif isinstance(train_dataset, Iterable):
            train_dataset = tuple(train_dataset)

        if tf.debugging.is_numeric_tensor(val_dataset) or (
                isinstance(val_dataset, np.ndarray)):
            val_dataset = (val_dataset,)
        elif isinstance(val_dataset, Iterable):
            val_dataset = tuple(val_dataset)

        if with_behaviour:
            assert len(train_dataset) > 1, ValueError(
                'The train dataset must be a list containing two elements: neural activity and behaviour')
            neural_dims = train_dataset[0].shape[1:]
            behavioural_dims = train_dataset[1].shape[1:]
            train_neural, train_behaviour = train_dataset[:2]
            if val_dataset is not None:
                if len(val_dataset) > 1:
                    assert neural_dims == val_dataset[0].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    assert behavioural_dims == val_dataset[1].shape[1:], ValueError(
                        'Validation and training datasets must have coherent sizes')
                    valid = val_dataset[:2]
        else:
            assert len(train_dataset) > 0, ValueError(
                'Please provide a non-empty train dataset')
            neural_dims = train_dataset[0].shape[1:]
            train_neural, train_behaviour = train_dataset[0], None
            if val_dataset is not None:
                assert neural_dims == val_dataset[0].shape[1:], ValueError(
                    'Validation and training datasets must have coherent sizes')
                valid = (val_dataset[0], None)

        return (train_neural, train_behaviour), valid

    @staticmethod
    def train(model_type: Union[str, ModelType], model_settings: Dict[str, Any], optimizer: tf.optimizers.Optimizer, epochs: int,
              train_dataset: Tuple[tf.Tensor, tf.Tensor], adaptive_weights: AdaptiveWeights,
              val_dataset: Optional[Tuple[tf.Tensor, tf.Tensor]] = None, batch_size: Optional[int] = None, logdir: Optional[str] = None,
              adaptive_lr: Optional[Union[dict, tf.keras.callbacks.Callback]] = None, layers_settings: Dict[str, Any] = {}):

        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        if layers_settings is None:
            layers_settings = {}

        (x, y), validation_data = Runtime.clean_datasets(
            train_dataset, val_dataset, model_type.with_behaviour)

        if model_type == ModelType.TNDM:
            model = TNDM(
                neural_space=x.shape[-1],
                behavioural_space=y.shape[-1],
                **model_settings,
                layers=layers_settings
            )
        elif model_type == ModelType.LFADS:
            model = LFADS(
                neural_space=x.shape[-1],
                **model_settings,
                layers=layers_settings
            )
        else:
            raise NotImplementedError(
                'This model type has not been implemented yet')

        callbacks = [adaptive_weights]
        if logdir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir))
        if adaptive_lr is not None:
            if isinstance(adaptive_lr, tf.keras.callbacks.Callback):
                callbacks.append(adaptive_lr)
            else:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', **adaptive_lr))

        model.build(input_shape=[None] + list(x.shape[1:]))

        model.compile(
            optimizer=optimizer,
            loss_weights=adaptive_weights.w
        )

        try:
            history = model.fit(
                x=x,
                y=y,
                callbacks=callbacks,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data
            )
        except KeyboardInterrupt:
            return model, None

        return model, history

    @staticmethod
    def train_from_file(settings_path: str):
        start_time=datetime.utcnow()

        is_json = settings_path.split('.')[-1].lower() == 'json'
        is_yaml = settings_path.split('.')[-1].lower() in ['yaml', 'yml']
        if is_json:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        elif is_yaml:
            with open(settings_path, 'r') as f:
                try:
                    settings = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)
        logger.info('Loaded settings file:\n%s' % yaml.dump(
            settings, default_flow_style=None, default_style=''))

        if 'seed' in settings.keys():
            seed = settings['seed']
        else:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info('Seed was set to %d' % (seed))

        model_type, model_settings, layers_settings, data, dataset_settings, runtime_settings, output_directory = Runtime.parse_settings(
            settings)
        (d_n_train, d_n_validation, _), (d_b_train, d_b_validation, _), _ = data
        optimizer, adaptive_weights, adaptive_lr, epochs, batch_size = runtime_settings
        logger.info('Arguments parsed')

        model, history = Runtime.train(
            model_type=model_type,
            model_settings=model_settings,
            layers_settings=layers_settings,
            train_dataset=(d_n_train, d_b_train),
            val_dataset=(d_n_validation, d_b_validation),
            optimizer=optimizer,
            adaptive_weights=adaptive_weights,
            adaptive_lr=adaptive_lr,
            epochs=epochs,
            batch_size=batch_size,
            logdir=os.path.join(output_directory, 'logs'))
        logger.info('Model training finished, now saving weights')

        model.save(os.path.join(output_directory, 'weights'))
        logger.info('Weights saved, now saving metrics history')

        pd.DataFrame(history.history).to_csv(os.path.join(output_directory, 'history.csv'))
        logger.info('Metrics history saved, now evaluating the model')

        stats = Runtime.evaluate_model(data, model)
        with open(os.path.join(output_directory, 'performance.json'), 'w') as fp:
            json.dump(stats, fp, cls=CustomEncoder, indent=2)
        logger.info('Model evaluated, now saving settings')

        end_time=datetime.utcnow()
        settings = dict(
            model=model_settings,
            dataset=dataset_settings,
            default_layers_settings=layers_settings.default_factory(),
            layers_settings=layers_settings,
            seed=seed,
            commit_hash=os.popen('git rev-parse HEAD').read().rstrip(),
            start_time=start_time.strftime('%Y-%m-%d %H:%M:%S.%f%Z'),
            end_time=end_time.strftime('%Y-%m-%d %H:%M:%S.%f%Z'),
            elapsed_time=str(end_time-start_time),
            author=getpass.getuser(),
            machine=socket.gethostname(),
            cpu_only_flag=ArgsParser.get_or_default(dict(os.environ), 'CPU_ONLY', 'FALSE') == 'TRUE',
            visible_devices=tf.config.get_visible_devices()
        )
        with open(os.path.join(output_directory, 'metadata.json'), 'w') as fp:
            json.dump(settings, fp, cls=CustomEncoder, indent=2)
        logger.info('Settings saved, execution terminated')

    @staticmethod
    def evaluate_model(data, model: tf.keras.Model):
        (d_n_train, d_n_validation, d_n_test), (d_b_train, d_b_validation, d_b_test), (d_l_train, d_l_validation, d_l_test) = data
        train_stats, ridge_model = Runtime.evaluate_performance(model, d_n_train, d_b_train, d_l_train)
        validation_stats, _ = Runtime.evaluate_performance(model, d_n_validation, d_b_validation, d_l_validation, ridge_model)
        test_stats, _ = Runtime.evaluate_performance(model, d_n_test, d_b_test, d_l_test, ridge_model)
        return dict(
            train=train_stats,
            validation=validation_stats,
            test=test_stats
        )
    
    def evaluate_performance(model: tf.keras.Model, neural: tf.Tensor, behaviour: tf.Tensor, latent: tf.Tensor, ridge_model=None):
        if isinstance(model, TNDM):
            log_f, b, (g0_r, mean_r, logvar_r), (g0_r, mean_i, logvar_i), (z_r, z_i), inputs = model(neural)
            z = np.concatenate([z_r.numpy().T, z_i.numpy().T], axis=0).T
        elif isinstance(model, LFADS):
            log_f, (g0, mean, logvar), z, inputs = model(neural)
        else:
            raise ValueError('Model not recognized')

        # Behaviour likelihood
        if model.with_behaviour:
            loss_fun = lnl.gaussian_loglike_loss(model.behaviour_sigma, [])
            b_like = loss_fun(behaviour, b).numpy() / behaviour.shape[0]
        else:
            b_like = None

        # Neural likelihood
        loss_fun = lnl.poisson_loglike_loss(model.timestep, ([0], [1]))
        n_like = loss_fun(None, (log_f, inputs)).numpy() / inputs.shape[0]

        # Behaviour R2
        if model.with_behaviour:
            unexplained_error = tf.reduce_sum(tf.square(behaviour - b)).numpy()
            total_error = tf.reduce_sum(tf.square(behaviour - tf.reduce_mean(behaviour, axis=[0,1]))).numpy()
            b_r2 = 1 - (unexplained_error / (total_error + 1e-10))
        else:
            b_r2 = None

        # Latent R2
        z_unsrt = z.T.reshape(z.T.shape[0], z.T.shape[1] * z.T.shape[2]).T
        l = latent.T.reshape(latent.T.shape[0], latent.T.shape[1] * latent.T.shape[2]).T
        if ridge_model is None:
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(z_unsrt, l)
        z_srt = ridge_model.predict(z_unsrt)
        unexplained_error = tf.reduce_sum(tf.square(l - z_srt)).numpy()
        total_error = tf.reduce_sum(tf.square(l - tf.reduce_mean(l, axis=[0,1]))).numpy()
        l_r2 = 1 - (unexplained_error / (total_error + 1e-10))
        return dict(
            behaviour_likelihood=b_like, 
            neural_likelihood=n_like,
            behaviour_r2=b_r2,
            neural_r2=l_r2), ridge_model

    @staticmethod
    def parse_settings(settings: Dict[str, Any]):
        # MODEL
        model = ArgsParser.get_or_error(settings, 'model')
        model_type: ModelType = ModelType.from_string(
            ArgsParser.get_or_error(model, 'type'))
        model_settings, layers_settings = Runtime.parse_model_settings(
            ArgsParser.get_or_error(model, 'settings'))

        # OUTPUT
        output = ArgsParser.get_or_error(settings, 'output')
        output_directory = ArgsParser.get_or_error(output, 'directory')

        # DATA
        data = ArgsParser.get_or_error(settings, 'data')
        data_directory = ArgsParser.get_or_error(data, 'directory')
        data_dataset_filename = ArgsParser.get_or_default(
            data, 'dataset_filename', 'dataset.h5')
        data_metadata_filename = ArgsParser.get_or_default(
            data, 'metadata_filename', 'metadata.json')
        dataset, dataset_settings = DataManager.load_dataset(
            directory=data_directory,
            filename=data_dataset_filename,
            metadata_filename=data_metadata_filename)
        neural_keys = ArgsParser.get_or_error(data, 'neural_keys')
        behavioural_keys = ArgsParser.get_or_default(
            data, 'behavioural_keys', {})
        latent_keys = ArgsParser.get_or_default(data, 'latent_keys', {})
        d_n_train, d_n_validation, d_n_test = Runtime.parse_data(
            dataset, neural_keys)
        d_b_train, d_b_validation, d_b_test = Runtime.parse_data(
            dataset, behavioural_keys)
        d_l_train, d_l_validation, d_l_test = Runtime.parse_data(
            dataset, latent_keys)
        valid_available = (d_b_validation is not None) and (
            d_n_validation is not None) if model_type.with_behaviour else (d_n_validation is not None)
        data = (
            (d_n_train, d_n_validation, d_n_test),
            (d_b_train, d_b_validation, d_b_test),
            (d_l_train, d_l_validation, d_l_test)
        )

        # RUNTIME
        runtime = ArgsParser.get_or_default(settings, 'runtime', {})
        initial_lr, lr_callback = Runtime.parse_learning_rate(
            ArgsParser.get_or_default(runtime, 'learning_rate', {}), valid_available)
        optimizer = Runtime.parse_optimizer(
            ArgsParser.get_or_default(runtime, 'optimizer', {}), initial_lr)
        weights = Runtime.parse_weights(
            ArgsParser.get_or_default(runtime, 'weights', {}), model_type)
        epochs = ArgsParser.get_or_default(runtime, 'epochs', 1000)
        batch_size = ArgsParser.get_or_default(runtime, 'batch_size', 8)
        runtime_settings = optimizer, weights, lr_callback, epochs, batch_size

        return model_type, model_settings, layers_settings, data, dataset_settings, runtime_settings, output_directory

    @staticmethod
    def parse_model_settings(model_settings):
        # DEFAULTS
        default_layer = ArgsParser.get_or_default_and_remove(
            model_settings, 'default_layer_settings', {})
        default_init = Runtime.parse_initializer(
            ArgsParser.get_or_default(default_layer, 'kernel_initializer', {}))
        default_reg = Runtime.parse_regularizer(
            ArgsParser.get_or_default(default_layer, 'kernel_regularizer', {}))

        # ALL OTHER LAYERS
        layers = defaultdict(
            lambda: {'kernel_regularizer': default_reg, 'kernel_initializer': default_init})
        custom_layers = ArgsParser.get_or_default_and_remove(
            model_settings, 'layers', {})
        for layer_name, settings in custom_layers.items():
            tmp_layer = deepcopy(layers.default_factory())
            for key, value in settings.items():
                if 'initializer' in key.lower():
                    tmp_layer[key] = Runtime.parse_initializer(value)
                elif 'regularizer' in key.lower():
                    tmp_layer[key] = Runtime.parse_regularizer(value)
                else:
                    tmp_layer[key] = value
            layers[layer_name] = tmp_layer

        return model_settings, layers

    @staticmethod
    def parse_initializer(settings):
        kernel_init_type = ArgsParser.get_or_default(settings, 'type', {})
        kernel_init_kwargs = ArgsParser.get_or_default(
            settings, 'arguments', {})
        if kernel_init_type.lower() == 'variance_scaling':
            i = tf.keras.initializers.VarianceScaling
        else:
            raise NotImplementedError(
                'Only variance_scaling has been implemented')
        return i(**kernel_init_kwargs)

    @staticmethod
    def parse_regularizer(settings):
        kernel_reg_type = ArgsParser.get_or_default(settings, 'type', {})
        kernel_reg_kwargs = ArgsParser.get_or_default(
            settings, 'arguments', {})
        if kernel_reg_type.lower() == 'l2':
            r = tf.keras.regularizers.L2
        else:
            raise NotImplementedError('Only l2 has been implemented')
        return r(**kernel_reg_kwargs)

    @staticmethod
    def parse_optimizer(optimizer_settings, initial_lr: float):
        type = ArgsParser.get_or_default(optimizer_settings, 'type', 'adam')
        kwargs = ArgsParser.get_or_default(optimizer_settings, 'arguments', {})
        if type.lower() == 'adam':
            opt = tf.keras.optimizers.Adam
        else:
            raise NotImplementedError(
                'Only Adam opimizer has been implemented')
        return opt(learning_rate=initial_lr, **kwargs)

    @staticmethod
    def parse_weights(weights_settings, model_type: ModelType):
        if weights_settings:
            initial = ArgsParser.get_or_default_and_remove(
                weights_settings, 'initial', [1.0])
            if not (len(initial) == model_type.weights_num):
                initial = [initial[0] for x in range(model_type.weights_num)]
            w = AdaptiveWeights(initial=initial, **weights_settings)
        else:
            w = AdaptiveWeights(
                initial=[1 for x in range(model_type.weights_num)])
        return w

    @staticmethod
    def parse_learning_rate(learning_rate_settings, valid_available: bool):
        initial_lr = ArgsParser.get_or_default_and_remove(
            learning_rate_settings, 'initial', 1e-2)
        monitor = 'val_loss' if valid_available else 'train_loss'
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, **learning_rate_settings)
        return initial_lr, lr_callback

    @staticmethod
    def parse_data(dataset: Dict[str, Any], keys: Dict[str, str]):
        fields = ['train', 'validation', 'test']
        out = []

        for field in fields:
            key = ArgsParser.get_or_default(keys, field, None)
            if (key is not None) and (key in (dataset.keys())):
                out.append(dataset[key].astype('float'))
            else:
                out.append(None)

        return out
