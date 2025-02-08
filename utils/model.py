from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Flatten


def build_linear_regression_model(window_size):
    """
    Builds a simple linear regression model using a Keras architecture.
    """
    input_layer = Input(shape=(window_size, 1), name="lr_input")
    flat = Flatten(name="flatten")(input_layer)
    output = Dense(1, activation="linear", name="lr_output")(flat)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_LSTM_model(window_size, lstm_units=50):
    """
    Builds a simple LSTM forecasting model using only the raw hourly time series data.
    """
    input_layer = Input(shape=(window_size, 1), name="input_layer")
    lstm_out = LSTM(lstm_units, return_sequences=False, name="simple_lstm")(input_layer)
    output = Dense(1, activation="linear", name="simple_output")(lstm_out)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_hybrid_model(window_size, num_fourier, lstm_units=50, dense_units=20):
    """
    Builds a hybrid forecasting model that combines an LSTM branch with a Fourier features branch.

    The LSTM branch processes the raw hourly time series and its output is passed through a Dense layer.
    In parallel, Fourier features are processed through another Dense layer.
    The outputs are concatenated and then fed through additional Dense layers to produce the final prediction.
    """
    power_input = Input(shape=(window_size, 1), name="power_input")
    fourier_input = Input(shape=(num_fourier,), name="fourier_input")

    lstm_out = LSTM(lstm_units, return_sequences=False, name="lstm_hidden")(power_input)
    dense_lstm = Dense(dense_units, activation="relu", name="dense_lstm")(lstm_out)

    dense_fourier = Dense(20, activation="relu", name="dense_fourier")(fourier_input)

    concat = Concatenate(name="concat")([dense_lstm, dense_fourier])

    fc1 = Dense(32, activation="relu", name="fc1")(concat)
    fc2 = Dense(16, activation="relu", name="fc2")(fc1)
    output = Dense(1, activation="linear", name="output")(fc2)

    model = Model(inputs=[power_input, fourier_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model
