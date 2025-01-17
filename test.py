
import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main():
    image_path = "bus.jpg"
    preprocessed_image = preprocess_image(image_path, target_size=(224, 224))
    print("Preprocessed image shape:", preprocessed_image.shape)
    
    interpreter = tf.lite.Interpreter(model_path="mobilenet_v2_35_quantized_int8.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_data = (preprocessed_image / 0.011) -1
    interpreter.set_tensor(input_details['index'], input_data.astype(np.int8))

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])[0]
    output_data = (output_data + 128) * 0.004
    print(output_data.shape)
    print(np.argmax(output_data))
    
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    predictions = model.predict(preprocessed_image)[0]
    print(predictions.shape)
    print(np.argmax(predictions))
    # decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)
    # for pred in decoded_predictions[0]:
    #     print(f"{pred[1]}: {pred[2]*100:.2f}%")

if __name__ == "__main__":
    main()
