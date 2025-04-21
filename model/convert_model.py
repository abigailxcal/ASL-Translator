import tensorflow as tf

# clear memory
tf.keras.backend.clear_session()
model = tf.keras.models.load_model("/Users/abigailcalderon/Downloads/asl_model.keras", compile=False)

model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter._experimental_lower_tensor_list_ops = False

# convert
try:
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Conversion complete. No drama.")
except Exception as e:
    print("Conversion failed:", e)