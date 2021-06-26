from json import JSONEncoder
import tensorflow as tf
import numpy as np


class CustomEncoder(JSONEncoder):
    def default(self, o):
        try:
            return super(CustomEncoder, self).default(o)
        except TypeError as e:
            if isinstance(o, tf.keras.regularizers.Regularizer) or isinstance(o, tf.keras.initializers.Initializer):
                return o.__dict__
            elif isinstance(o, np.ndarray):
                return o.tolist()
            else:
                return o.__dict__