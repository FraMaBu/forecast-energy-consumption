import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.decomposition import PCA
import numpy as np

class PCAInitializer(tf.keras.initializers.Initializer):
    """
    Custom initializer to set Dense layer weights based on PCA components.
    """
    def __init__(self, pca_weights):
        self.pca_weights = pca_weights

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.constant(self.pca_weights, dtype=dtype)

    def get_config(self):
        return {"pca_weights": self.pca_weights.tolist()}

def build_preliminary_model(window_size, lstm_units=50):
    """
    Builds a simple LSTM model that predicts the next price from a sequence.
    This model is used to extract hidden state representations.
    """
    price_input = Input(shape=(window_size, 1), name='price_input')
    lstm_layer = LSTM(lstm_units, return_sequences=False, name='lstm_hidden')(price_input)
    pred = Dense(1, name='price_pred')(lstm_layer)
    model = Model(inputs=price_input, outputs=pred)
    model.compile(optimizer='adam', loss='mse')
    return model

def compute_pca_weights(model, X_seq_train, n_components=20):
    """
    Given a trained preliminary model, extract the hidden states and compute PCA.
    Returns the PCA components (transposed) for initializing a Dense layer.
    """
    # Create a submodel that outputs the hidden state from the LSTM.
    hidden_model = Model(inputs=model.input, outputs=model.get_layer('lstm_hidden').output)
    hidden_states = hidden_model.predict(X_seq_train)
    
    pca = PCA(n_components=n_components)
    pca.fit(hidden_states)
    # pca.components_ has shape (n_components, lstm_units); we need shape (lstm_units, n_components)
    pca_weights = pca.components_.T
    return pca_weights

def build_hybrid_model(window_size, num_fourier, pca_weights, lstm_units=50, n_pca_components=20):
    """
    Builds the hybrid forecasting model that combines:
      - An LSTM branch (with a PCA-initialized Dense layer)
      - A branch with precomputed Fourier features.
    """
    # Inputs
    price_input_final = Input(shape=(window_size, 1), name='price_input_final')
    fourier_input = Input(shape=(num_fourier,), name='fourier_input')
    
    # LSTM branch
    lstm_out_final = LSTM(lstm_units, return_sequences=False, name='lstm_hidden_final')(price_input_final)
    
    # Dense layer initialized with PCA weights.
    pca_layer = Dense(n_pca_components,
                      activation='linear',
                      kernel_initializer=PCAInitializer(pca_weights),
                      bias_initializer='zeros',
                      name='pca_layer')(lstm_out_final)
    
    # Fourier branch
    fourier_dense = Dense(20, activation='relu', name='fourier_dense')(fourier_input)
    
    # Concatenate branches and add fully connected layers.
    concat = Concatenate(name='concat')([pca_layer, fourier_dense])
    fc1 = Dense(32, activation='relu', name='fc1')(concat)
    fc2 = Dense(16, activation='relu', name='fc2')(fc1)
    output = Dense(1, activation='linear', name='output')(fc2)
    
    model = Model(inputs=[price_input_final, fourier_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_simple_model(window_size, lstm_units=50):
    """
    Builds a simple LSTM forecasting model using only the raw price sequence.
    """
    simple_price_input = Input(shape=(window_size, 1), name='simple_price_input')
    simple_lstm = LSTM(lstm_units, return_sequences=False, name='simple_lstm')(simple_price_input)
    simple_output = Dense(1, activation='linear', name='simple_output')(simple_lstm)
    
    model = Model(inputs=simple_price_input, outputs=simple_output)
    model.compile(optimizer='adam', loss='mse')
    return model