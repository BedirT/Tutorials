# https://youtu.be/pAhPiF3yiXI
# Tensorflow Tutorial 2 - Tensor Basics
# - tensor initializing
# - math operations
# - indexing
# - reshaping

import tensorflow as tf

#############################
# initialization of tensors #
#############################
x = tf.constant(4) 
x = tf.constant(value=4., dtype=tf.float32, shape=(3, 1))

# Ones in the tensor in given shape
x = tf.ones((3, 3))

# Zeros in the tensor in given shape
x = tf.zeros((1, 3))

# Eye - Identity Matrix (I)
x = tf.eye(4)

# Normal Distributed Random Init
x = tf.random.normal((3,3), mean=0, stddev=1)

# Uniform Random DIstr.
x = tf.random.uniform((3,2), minval=0, maxval=1)

# Range of values - start to end - like python range()
x = tf.range(start=0, limit=10, delta=2)

# Type Casting
x = tf.cast(x, dtype=tf.float16)
# print(x)

###################
# Math Operations #
###################
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

# Element wise Addition
z = tf.add(x, y)
z = x + y

# Element wise subtriction
z = tf.subtract(x, y)
z = x - y

# Element wise multiplication
z = tf.multiply(x, y)
z = x * y

# Element wise division
z = tf.divide(x, y)
z = x / y

# Dot Product
z = tf.tensordot(x, y, axes=1)
z = tf.reduce_sum(x*y, axis=0)
print(z)

# Element wise exponantiate
z = x ** 5

# Matrix Multiplication
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x, y)
z = x @ y
print(z)

############
# Indexing #
############
# indexing is same as python lists
x = tf.range(1, 10)
print(x[:])
print(x[:-2])
print(x[2: 5])
print(x[::-1])

# gathering specific indexes .gather
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)

# specific indexes on multi-dims
x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
print(x[0,:])
print(x[0:2,:])

#############
# Reshaping #
#############
x = tf.range(9)
print(x)

# simple reshape
x = tf.reshape(x, (3,3))
print(x)

# simple transpose
x = tf.transpose(x, perm=[1,0])
print(x)

# exploring perm
# perm - specify the order of the dimensions
# [1,2,0] First dim, second dim, and 0th...
x = tf.reshape(tf.range(16), (2,2,4))
print(x)
x = tf.transpose(x, perm=[1,2,0])
print(x)