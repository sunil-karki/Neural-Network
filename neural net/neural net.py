import random
import numpy as np
import scipy.io as sc
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from tensorflow.examples.tutorials.mnist import input_data


#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
"""mat = sc.loadmat("C:\\Users\\Dell\\Downloads\\coursera\\machine-learning-coursera\\machine-learning-ex4\\machine-learning-ex4\\ex4-Neural\\ex4data1.mat")"""
mat = sc.loadmat("F:\\From_C\\Downloads\\coursera\\machine-learning-coursera\\machine-learning-ex4\\machine-learning-ex4\\ex4\\ex4data1.mat")
X = mat['X']


#We have to break one layer of array in mat['y'], So;
a = mat['y'].flatten()

#Here 10 means 0 and other as usual in data extracted from  'mat'.
#replace 10 by 0
a[a==10] = 0


"""Encoding the Labels in array(which is 'Y')"""
original_indices = tf.constant(a)
depth = tf.constant(10)
one_hot_encoded = tf.one_hot(indices=original_indices, depth=depth)

with tf.Session():
    Y = one_hot_encoded.eval()
    print("Data has been extracted:")



m_of_X, f_of_X = mat['X'].shape

n_nodes_hl1 = 25
#n_nodes_hl2 = 500
#n_nodes_hl3 = 500

"""no. of labels"""
n_classes = 10
"""Iteration, I suppose"""
batch_size = 100

"""20x20 pixels image = flat 400 pixels(features of each example in data)"""
x = tf.placeholder('float', [None, 400])
y = tf.placeholder('float')


"""
mse_history = []
accuracy_history = []
cost_history = np.empty(shape=[1], dtype=float)


"""


X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)


def display_data(array_mat):

    """reshaping the array back into 20x20 pixels for display"""
    img = array_mat.reshape(20,20)
    img = np.transpose(img) #arranging the pixel intensities(array)
    
    imgplot = plt.imshow(img, cmap=plt.cm.binary)
    #plt.show()
    plt.draw()
    plt.pause(0.001)


def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([400, n_nodes_hl1])),
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
    #l1 = tf.nn.relu(l1)  """Not accurate or perfect as sigmoid"""
    #l1 = tf.nn.softmax(l1)  """Not very good either"""
    l1 = tf.nn.sigmoid(l1)
                        
    """ l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #rectified linear: relu
    #l2 = tf.nn.relu(l2)
    """

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    train_step = tf.train.GradientDescentOptimizer(1).minimize(cost)

    #epoch === iterations
    
    with tf.Session() as sess:
        #old-version:  sess.run(tf.initialize_all_variables())
        #new-version:
        sess.run(tf.global_variables_initializer())

        for epoch in range(600):
                        
            #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            
            _, c = sess.run([train_step, cost], feed_dict = {x: train_x, y: train_y})
            """cost_history = np.append(cost_history, cost)"""
            print('Epoch', epoch, 'completed out of 100.  loss:', c)

            """correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            """


            """
            pred_y = sess.run(y, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)
            accuracy = (sess.run(accuracy, feed_dict={x: train_x, y: train_y}))
            accuracy_history.append(accuracy)

            print('epoch: ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "-Train accuracy:", accuracy)
            
        save_path = saver.save(sess, model_path)    
        print("Model saved in file: %s" %save_path)


        plt.plot(mse_history, 'r')
        plt.show()
        plt.plot(accuracy_history)
        plt.show()
            
            """


        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))    
        print("Accuracy: ", (sess.run(accuracy, feed_dict={x: train_x, y: train_y})) )



        print("Displaying Train Data:")
        plt.ion()
        plt.show()
        for i in range(m_of_X):
            arr = random.randint(1, 5000)
            print("random: ", arr)
            display_data(X[arr])
            
            print("Prediction: ", (sess.run(tf.argmax(prediction,1), feed_dict={x: [X[arr]]}) ) )
            print("Accuracy: ", (sess.run(accuracy, feed_dict={x: [X[arr]], y: [Y[arr]]})) )

            s = input("Paused; Enter 'e' & 'enter' to exit, 'enter' to continue: ")
            if s == 'e':
                break

        


train_neural_network(x)

