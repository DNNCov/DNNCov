# output_fetcher.py
# Generates fetch functions to retrieve outputs.
# see DeepHunter fuzzone.py
# TODO: functions can likely be moved into other files to make repository less convoluted

from keras import backend as K
import numpy as np
def predict(self, input_data):
    inp = self.model.input
    functor = K.function([inp] + [K.learning_phase()], self.outputs)
    outputs = functor([input_data, 0])
    return outputs
def fetch_function(handler, input_batches, preprocess):
    _, img_batches, _, _, _ = input_batches
    if len(img_batches) == 0:
        return None, None
    preprocessed = preprocess(img_batches)
    layer_outputs = handler.predict(preprocessed)
    # Return the prediction outputs
    return layer_outputs, np.expand_dims(np.argmax(layer_outputs[-1], axis=1),axis=0)

def build_fetch_function(handler, preprocess,models=None):
    def func(input_batches):
        """The fetch function."""
        if models is None:
            return fetch_function(
                handler,
                input_batches,
                preprocess
            )
    return func