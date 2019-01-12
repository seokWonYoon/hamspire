#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# In[2]:


# ******
# 신경망 모델 구성
# ******
# 입력값의 차원은 [배치크기, 특성값] 으로 되어 있음
# 손글씨 이미지는 28*28 픽셀로 이뤄져있음. 이를 784개의 feature  로 설정함

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10]) # 결과는 0 ~ 9 까지 10가지 분류


# In[3]:


# 신경망의 레이어 구성
# 784(입력 특성값)
# -> 256 (히든 레이어 뉴런 갯수) -> 256 
# -> 10 (결과값)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
# 입력값에 가중치 곱한후 ReLU 함수를 이용하여 레이어 생성
L1 = tf.nn.relu(tf.matmul(X, W1))
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
# L1에 가중치 곱한후 ReLU 함수를 이용하여 레이어 생성
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256,10], stddev=0.01))
# 최종 모델의 출력값은 W3 변수를 곱해 10개의 분류를 가지게 됨


# In[4]:


model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# In[5]:


# ********
# 신경망 모델 학습
# ********
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼의 학습데이터 호출
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
    print('Epoch:', '%0.4d'  % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
print('최적화 완료')


# In[6]:


# ******
# 결과 확인
# *****
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) # argmax 는 인자값 중 가장 큰 값을 취함
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy,
                      feed_dict={X: mnist.test.images,
                                 Y: mnist.test.labels}))


# In[ ]:





# In[ ]:




