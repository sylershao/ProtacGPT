import tensorflow as tf

state = tf.Variable(0.0,dtype=tf.float32)
one = tf.constant(1.0,dtype=tf.float32)
new_val = tf.add(state, one)
update = tf.assign(state, new_val)# 返回tensor， 值为new_val
update2 = tf.assign(state, 10000)# 没有fetch，便没有执行
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print(sess.run(update))