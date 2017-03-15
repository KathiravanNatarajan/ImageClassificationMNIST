from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensoflow as tf 

# parameters 
learnrate = 0.001
training_epochs = 20 
batch_size = 128 
display_step = 1 

n_input = 784 
n_classes = 10 

n_hiddenlayer = 256 

weights= {
          'hiddenlayer':tf.Variable(tf.random_normal([n_input,n_hiddenlayer]))
          'output':tf.Variable(tf.random_normal([n_hiddenlayer,n_classes])) 
          }
biases = {
          'hiddenlayer':tf.Variable(tf.random_normal([n_hidden]))
          'output':tf.Variable(tf.random_normal([n_classes]))
          }
x = tf.placeholder("float",[None, 28,28,1])
y = tf.placeholder("float",[None, n_classes])

x_flat = tf.reshape(x,[-1,n_input])

#model

hiddeninput = tf.add(tf.matmul(x_flat,weights['hiddenlayer']),biases['hiddenlayer'])
hiddenoutput = tf.nn.relu(hiddeninput)
logits = tf.add(tf.matmul(hiddenoutput,weights['output']),biases['output'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_intializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})