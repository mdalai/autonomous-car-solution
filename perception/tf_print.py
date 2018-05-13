import tensorflow as tf

def run():
    # create a Graph
    g = tf.Graph()
    with g.as_default():
      a = tf.constant(value=1.0, dtype=tf.float32, name='a')
      b = tf.constant(value=5.0, dtype=tf.float32, name='b')
      c = tf.constant(value=2.0, dtype=tf.float32, name='c')
      d = tf.constant(value=3.0, dtype=tf.float32, name='d')
      m1 = tf.multiply(a,b)
      print("m1: ", m1)
      m1_output = tf.Print(m1,[m1,b,a,d],"Begin to TF Print m1: ")
      m2 = tf.multiply(c,d)
      print("m2: ",m2)
      m2_output = tf.Print(m2,[m2,c,d],"Begin to TF Print m2: ")
      m3 = tf.add(m1_output,m2_output)
      print("m3: ",m3)  

    # run
    with tf.Session(graph=g) as sess:
      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess.run(init_op)
      output = sess.run(m3)
      print(output)


if __name__ == '__main__':
  run()

