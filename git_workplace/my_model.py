# EEG-TCN NET
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.models import Model
import keras.backend as K
from tensorflow.keras.constraints import max_norm

# Temporal Convolutional Networks(TCN)
def residual_block(x, filters = 128, kernel_size = 3, dilation_rate = 1):
    # Residual
    x_res = x
    # ConvNet
    x_out = WeightNormalization(layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'causal'))(x)
    x_out = layers.Activation('elu')(x_out)
    x_out = WeightNormalization(layers.Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, padding = 'causal'))(x_out)
    x_out = layers.Activation('elu')(x_out)
    # Residual Outputs
    residual = layers.Add()([x_res, x_out])
    residual = layers.Activation('elu')(residual)
    
    return residual, x_out

def EEGNet(input_len = 4000, n_channel = 1, kernLength = 64, F1 = 32, D = 2, F2 = 64, norm_rate = 0.25, n_filters = 128):
    
    # Reshape (timestep = 4000, 1) to (channel = 1, timestep = 4000, 1)
    inputs = layers.Input(shape = (input_len, n_channel))
    reshape = layers.Reshape((n_channel, input_len, 1))(inputs)
    # EEG Net
    block1 = layers.Conv2D(F1, (1, kernLength), padding = 'same', use_bias = False)(reshape)
    block1 = layers.BatchNormalization()(block1)
    
    block1 = layers.DepthwiseConv2D((n_channel, 1), use_bias = False, depth_multiplier = D, depthwise_constraint = max_norm(1.))(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    
    block2 = layers.SeparableConv2D(F2, (1, 16), use_bias = False, padding = 'same')(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.Reshape((input_len, F2))(block2)
    block2 = layers.Conv1D(filters = F2, kernel_size = 7, strides = 2, padding = 'same')(block2)
    
    # TCN Net
    tcn = layers.Conv1D(filters = n_filters, kernel_size = 1, padding = 'same')(block2)
    tcn, s1 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 1)
    tcn, s2 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 2)
    tcn, s3 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 4)
    tcn, s4 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 8)
    tcn, s5 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 16)
    tcn, s6 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 32)
    tcn, s7 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 64)
    tcn, s8 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 128)
    tcn, s9 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 256)
    tcn, s10 = residual_block(tcn, filters = n_filters, kernel_size = 3, dilation_rate = 512)
    skip_connection = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
    skip_connection = layers.Add()(skip_connection)
    skip_connection = layers.Activation('elu')(skip_connection)
    outputs = layers.GlobalAveragePooling1D()(skip_connection)
    # Classifier
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(n_filters, activation = 'elu')(outputs)
    outputs = layers.Dense(1, activation = 'sigmoid')(outputs)
    # Model compile
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam( learning_rate = 0.0005 )
    auc = tf.keras.metrics.AUC(name = 'auc')
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = [auc])

    return model