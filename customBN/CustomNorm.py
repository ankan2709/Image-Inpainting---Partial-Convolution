import tensorflow.keras.backend as K
from tensorflow.keras import layers
from customBN.custom_objects import ANInitializer



class CustomNorm(layers.BatchNormalization):
  def __init__(self, n_mixture=5, momentum=0.99, epsilon=0.1, axis=-1, **kwargs):
    super(CustomNorm, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
    self.n_mixture = n_mixture
    # self.axis = axis

    if self.axis == -1:
            self.data_format = 'channels_last'
    else:
        self.data_format = 'channel_first'

  def build(self, input_shape):

    super(CustomNorm, self).build(input_shape)

    ndims = len(input_shape)
    dim = input_shape[-1]   # self.axis
    shape = (self.n_mixture, dim) # K x C 
    
    self.FC = layers.Dense(self.n_mixture, activation="sigmoid")
    self.FC.build(input_shape) # (N, C)
    
    if len(input_shape) == 4:
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
    else:
        self.GlobalAvgPooling = layers.GlobalAveragePooling1D(self.data_format)
    self.GlobalAvgPooling.build(input_shape)
    
    self._trainable_weights = self.FC.trainable_weights
    
    self.learnable_weights = self.add_weight(name='gamma2', 
                                  shape=shape,
                                  initializer=ANInitializer(scale=0.1, bias=1.),
                                  trainable=True)

    self.learnable_bias = self.add_weight(name='bias2', 
                                shape=shape,
                                initializer=ANInitializer(scale=0.1, bias=0.),
                                trainable=True)

  def call(self, inputs):
    
    input_shape = K.int_shape(inputs)
    ndims = len(input_shape)

    avg = self.GlobalAvgPooling(inputs) # N x C 
    attention = self.FC(avg) # N x K 
    gamma_readjust = K.dot(attention, self.learnable_weights) # N x C
    beta_readjust  = K.dot(attention, self.learnable_bias)
    val = super(CustomNorm, self).call(inputs)

    # # broadcast if needed
    if K.int_shape(inputs)[0] is None or K.int_shape(inputs)[0] > 1:
        if len(input_shape) == 4:
            gamma_readjust = gamma_readjust[:, None, None, :]
            beta_readjust  = beta_readjust[:, None, None, :]
        else:
            gamma_readjust = gamma_readjust[:, None, :]
            beta_readjust  = beta_readjust[:, None, :]

    return gamma_readjust * val + beta_readjust


    def get_config(self):
        config = {
            'n_mixture' : self.n_mixture
        }
        base_config = super(AttentiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))