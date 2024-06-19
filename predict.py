import cv2
import numpy as np
import tensorflow as tf


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img_flatten = img.flatten()
    img_ready = ((img_flatten / 255.) - 0.5) * 2
    img_ready = np.expand_dims(img_ready, axis=0)
    return img_ready


if __name__ == '__main__':
    img_path = "img_sample/8.png"

    # reload the model
    trained_model = tf.keras.models.load_model(filepath="record/my_best_model.keras")

    # preprocess the test image
    processed_image = preprocess_image(img_path)
    result = trained_model.predict(processed_image)

    # analysis the result
    predicted_label = np.argmax(result, axis=1)[0]
    confidence = np.max(result, axis=1)[0]

    print(f"result: {predicted_label}")
    print(f"prob: {confidence:.4f}")
