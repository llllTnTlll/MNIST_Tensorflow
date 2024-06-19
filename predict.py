import cv2
import numpy as np
import tensorflow as tf


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img_flatten = img.flatten()
    img_ready = ((img_flatten / 255.) - 0.5) * 2  # 归一化到[-1, 1]的范围内
    img_ready = np.expand_dims(img_ready, axis=0)
    return img_ready


if __name__ == '__main__':
    img_path = "img_sample/8.png"

    # 构建模型并加载权重
    trained_model = tf.keras.models.load_model(filepath="record/my_best_model.keras")

    # 预处理图像并进行预测
    processed_image = preprocess_image(img_path)
    result = trained_model.predict(processed_image)

    # 解析预测结果
    predicted_label = np.argmax(result, axis=1)[0]
    confidence = np.max(result, axis=1)[0]

    print(f"预测结果: {predicted_label}")
    print(f"置信度: {confidence:.4f}")