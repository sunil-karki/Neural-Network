from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
digits = datasets.load_digits()
#print(digits.data.shape)      #(1797, 64)
print(digits.data[1])
"""

"""
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(X.shape)
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x = mnist.train.images  #(55000, 784)
train_y = mnist.train.labels
test_x = mnist.test.images    #(10000, 784)
test_y = mnist.test.labels  



def display_data(array_mat):

    """reshaping the array back into 28x28 pixels for display"""
    img = array_mat.reshape(28,28)
    #img = np.transpose(img) #arranging the pixel intensities(array)
    
    imgplot = plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()
    plt.draw()
    plt.pause(0.001)



def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    """hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    """
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #rectified linear: relu
    #l1 = tf.nn.relu(l1)  
    #l1 = tf.nn.softmax(l1)  
    l1 = tf.nn.sigmoid(l1)
                        
    """ l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #rectified linear: relu
    #l2 = tf.nn.relu(l2)
    """

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    #train_step = tf.train.GradientDescentOptimizer(1).minimize(cost)
    train_step = tf.train.AdamOptimizer().minimize(cost)

    
    with tf.Session() as sess:
        #old-version:  sess.run(tf.initialize_all_variables())
        #new-version:
        sess.run(tf.global_variables_initializer())

        for epoch in range(400):
                        
            #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            
            _, c = sess.run([train_step, cost], feed_dict = {x: train_x, y: train_y})
            """cost_history = np.append(cost_history, cost)"""
            print('Epoch', epoch, 'completed out of 100.  loss:', c)

         
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))    
        print("Accuracy: ", (sess.run(accuracy, feed_dict={x: train_x, y: train_y})) )



        print("Displaying Train Data:")
        plt.ion()
        plt.show()
        for i in range(row_X):
            arr = random.randint(1, 980)
            print("random: ", arr)
            display_data(test_x[arr])
            
            print("Prediction: ", (sess.run(tf.argmax(prediction,1), feed_dict={x: [test_x[arr]]}) ) )
            print("Accuracy: ", (sess.run(accuracy, feed_dict={x: [test_x[arr]], y: [test_y[arr]]})) )

            s = input("Paused; Enter 'e' & 'enter' to exit, 'enter' to continue: ")
            if s == 'e':
                break



#display_data(digits.data[9])




n_nodes_hl1 = 500
#n_nodes_hl2 = 500
#n_nodes_hl3 = 500

"""no. of labels"""
n_classes = 10

batch_size = 100

"""20x20 pixels image = flat 400 pixels(features of each example in data)"""
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

train_x, train_y = shuffle(train_x, train_y, random_state=1)
test_x, test_y = shuffle(test_x, test_y, random_state=1)

"""Taking only few examples for training"""
train_x = train_x[:8000]
train_y = train_y[:8000]
test_x = test_x[:1000]
test_y = test_y[:1000]

row_X, col_X = train_x.shape

train_neural_network(x)

    
