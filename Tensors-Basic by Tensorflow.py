

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf



#Initialization
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.random.normal((3,3), mean=10, stddev=1)
x = tf.eye(4)
print(x)


x = tf.constant([1,2,3])
y = tf.constant([8,7,6])

#Math operation
z1 = x + y
z2 = x - y
z3 = x * y  # dot product
z4 = x / y
z = x**y

print(z1, z2, z3, z4, z)


z11 = tf.add(x, y)
z22 = tf.subtract(x, y)
z33 = tf.multiply(x, y)  # dot product
z44 = tf.divide(x, y)

print(z11, z22, z33, z44)

z5 = tf.tensordot(x, y, axes=1)
z55 = tf.reduce_sum(x*y, axis=0)  #same before operation

print(z5, z55)


xmat = tf.random.normal((2,3))
ymat = tf.random.normal((3,5))
zmat = tf.matmul(xmat, ymat)
print(zmat)

zmat2 = xmat @ ymat  #same matmul operation
print(zmat2)

#Indexing
xi = tf.constant([1,2,1,2,3,2,1,0,4,1])
print(xi)
print(xi[:])
print(xi[:-1])
print(xi[:-2])
print(xi[3:-3])
print(xi[::2])
print(xi[::-1])


indices = tf.constant([0,4])
x_indi = tf.gather(xi, indices)
print(x_indi)

#Indexing for matrix
xmat2 = tf.constant([[1,2,3],[4,3,7],[0,9,3]])
print(xmat2[0:-1])


#Reshape
xr = tf.range(9)
print(xr)

xr = tf.reshape(xr, (3,3))
print(xr)

xr = tf.transpose(xr, perm=[1,0])
print(xr)



#Thanks to Dino Persson


