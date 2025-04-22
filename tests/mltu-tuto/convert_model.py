# import tensorflow as tf
# import tf2onnx
# import onnx
# from tensorflow import keras
# from keras.models import load_model

# from keras.layers import Lambda
# import keras.backend as K

# model = load_model('Models/05_sound_to_text/202503171609/model.h5')

# onnx_model, _ = tf2onnx.convert.from_keras(model)
# onnx.save(onnx_model, 'Models/05_sound_to_text/202503171609/model-converted.onnx')

###############

import tensorflow as tf
import tf2onnx

# Path to your H5 model
h5_model_path = "Models/05_sound_to_text/202503171609/model.h5"
# Path where you want to save the ONNX model
onnx_model_path = h5_model_path.replace(".h5", ".onnx")

# Create a new model that wraps your existing model
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        
    def call(self, inputs):
        # Define your model's forward pass here
        # This should match what your model does
        # For example:
        x = tf.transpose(inputs, perm=[0, 2, 1])
        # Add more operations as needed
        return x

# Create and save a new model
custom_model = CustomModel()
# Define input shape matching your model
dummy_input = tf.random.normal((1, 1392, 193))
output = custom_model(dummy_input)

# Convert to ONNX
input_signature = [tf.TensorSpec((None, 1392, 193), tf.float32, name="input")]
model_proto, _ = tf2onnx.convert.from_function(
    custom_model,
    input_signature=input_signature,
    opset=13,
    output_path=onnx_model_path
)

print(f"Model successfully converted and saved to {onnx_model_path}")