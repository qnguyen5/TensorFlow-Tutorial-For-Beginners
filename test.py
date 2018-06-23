import tensorflow as tf
import numpy as np
import os

"""
#Set reproducible experiment
np.random.seed(0)

x1 = [np.random.rand() for _ in range(4)]
x2 = [np.random.rand() for _ in range(4)]

print("x1 = ",x1)
print("x2 = ",x2)

x1 = tf.constant(x1)
x2 = tf.constant(x2)

#multiply
x1multx2 = tf.multiply(x1,x2)
x2multx1  = tf.multiply(x2,x1)

#initialize the sesion
sess = tf.Session()
config = tf.ConfigProto(log_device_placement = True)

#print the result
print('x1multx2 = ',sess.run(x1multx2))
print('x2multx1 =',sess.run(x2multx1))

#close the session
sess.close()
"""
