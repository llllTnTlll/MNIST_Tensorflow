# Handwriting number recognition based on Tensorflow
**This is the example program I used in the presentation conducted on June 21st.** 
In this sample program, I used tensorflow to build small neural networks and complete training on the MNIST dataset. 
Since the reference book is so old that most of the api's have been removed by tensorflow, I used the low level api's that are as close as possible to the original book and modified them based on that.

## DatasetStructure
> **data for tarin**  
>>train-images.idx3-ubyte  
>>train-labels.idx1-ubyte

> **data for test**  
>> t10k-images.idx3-ubyte  
>> t10k-labels.idx1-ubyte  


## AutoGraph
```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def train_step(data, label):
    with tf.GradientTape() as tape:
        logits = my_model(data)
        loss = loss_fn(label, logits)
    gradients = tape.gradient(loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    return loss
```

In the era of TensorFlow 1.0, a static computation graph was employed. It required the creation of the computation graph using various TensorFlow operators first, followed by initiating a session explicitly to execute the graph.

However, in the era of TensorFlow 2.0, a dynamic computation graph is adopted. After each operator is used, it is dynamically added to the implicit default computation graph and executed immediately to obtain results without the need to initiate a session. 

If there is a need to use a static graph in TensorFlow 2.0, the @tf.function decorator can be used to convert regular Python functions into corresponding TensorFlow computation graph construction code. Running such a function is equivalent to executing code using a session in TensorFlow 1.0. The approach of building static graphs using tf.function is called Autograph.
