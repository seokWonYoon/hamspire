
# 털과 날개의 유무에 따라 포유류와 조류를 분류하는 신경망 모델
import tensorflow as tf
import numpy as np
x_data = np.array(
    [[0,0],
     [1,0],
     [1,1],
     [0,0],
     [0,0],
     [0,1]])
# [기타, 포유류, 조류]

y_data = np.array([
    [1,0,0], # 기타
    [0,1,0], # 포유류
    [0,0,1], # 조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
    ])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.plasceholder()