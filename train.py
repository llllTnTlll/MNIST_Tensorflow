import tensorflow as tf
import model
import data
from tqdm import tqdm


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def train_step(data, label):
    with tf.GradientTape() as tape:
        logits = my_model(data)
        loss = loss_fn(label, logits)
    gradients = tape.gradient(loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    return loss


if __name__ == "__main__":
    # =======================TO BUILD AND INITIAL MODEL===============================
    # build model
    my_model = model.build_model()
    # loss function & optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    my_model.compile(optimizer=optimizer, loss=loss_fn)
    # =======================TO TRAIN THIS MODEL======================================
    dataset_path = './dataset'
    best_model_path = 'record/my_best_model.keras'
    # my_model.load_weights(best_model_path)

    images, labels = data.load_mnist(dataset_path)

    best_loss = float('inf')
    for epoch in range(50):
        batch_generator = data.create_batch_generator(images, labels, shuffle=True)
        batch_training_losses = []
        for batch_x, batch_y in batch_generator:
            loss = train_step(batch_x, batch_y)
            batch_training_losses.append(loss)

        avg_loss = sum(batch_training_losses)/len(batch_training_losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            my_model.save(best_model_path)

        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")



