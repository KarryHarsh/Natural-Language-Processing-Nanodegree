from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    drpout = Dropout(0.3)(bn_cnn)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(drpout)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers,rnn_type = GRU, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    prev_layer = input_data
    for i in range(recur_layers):
        prev_layer = rnn_type(units, activation='relu', return_sequences=True, implementation=2, name=f'rnn_{i+1}')(prev_layer)
        prev_layer = BatchNormalization()(prev_layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(prev_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn1 = Bidirectional(GRU(units, activation='relu', return_sequences=True))(input_data)
    bnn_rnn1 =BatchNormalization()(bidir_rnn1)
    drpout = Dropout(0.4)(bnn_rnn1)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(drpout)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def self_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    input_dim: Input dimension of the data.
    filters: the dimensionality of the output space
    kernel_size:  An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    conv_stride: An integer or tuple/list of a single integer, specifying the stride length of the convolution
    conv_border_mode: Type of padding.
    units:  Positive integer, dimensionality of the output space.
    output_dim: output sequence dimension
    recur_layers: No of RNN layer.
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    drpout1 = Dropout(0.3)(bn_cnn)
    # bidirectional recurrent layer
    bidir_rnn1 = Bidirectional(GRU(units, activation='relu', return_sequences=True))(drpout1)
    bnn_rnn1 =BatchNormalization()(bidir_rnn1)
    drpout2 = Dropout(0.3)(bnn_rnn1)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(drpout2)
    drpout3 = Dropout(0.3)(time_dense)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(drpout3)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29,recur_layers = 3,cnn_layers = 1,dropout = 0.3, rnn_type = GRU ):
    """ Build a deep neural network model for speech Recognizatuin.
    input_dim: Input dimension of the data.
    filters: the dimensionality of the output space
    kernel_size:  An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    conv_stride: An integer or tuple/list of a single integer, specifying the stride length of the convolution
    conv_border_mode: Type of padding.
    units:  Positive integer, dimensionality of the output space.
    output_dim: output sequence dimension
    recur_layers: No of RNN layer.
    cnn_layers: No of CNN layer.
    dropout: Fractional value droupout range between 0 to 1.
    rnn_type = Type of RNN layer. e.g. SimpleRNN, LSTM,  GRU
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Specifying the layers in Neural network
    # Add convolutional layer, each with batch normalization
    prev_cnn_layer = input_data
    for i in range(cnn_layers):
        prev_cnn_layer = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name=f'conv1d_{i+1}')(prev_cnn_layer)
        prev_cnn_layer = BatchNormalization(name=f'bn_conv_{i+1}')(prev_cnn_layer)
        prev_cnn_layer = Dropout(dropout)(prev_cnn_layer)
    # TODO: Add recurrent layers, each with batch normalization
    prev_rnn_layer = prev_cnn_layer
    for i in range(recur_layers):
        prev_rnn_layer = Bidirectional(rnn_type(units, activation='relu', return_sequences=True,name=f'rnn_{i+1}'))(prev_rnn_layer)
        prev_rnn_layer = BatchNormalization()(prev_rnn_layer)
        prev_rnn_layer = Dropout(dropout)(prev_rnn_layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense1 = TimeDistributed(Dense(output_dim))(prev_rnn_layer)
    drpout1 = Dropout(dropout)(time_dense1)
    time_dense = TimeDistributed(Dense(output_dim))(drpout1)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    #summarizing the model architecture.
    print(model.summary())
    return model