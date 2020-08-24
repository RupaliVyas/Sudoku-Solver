import tensorflow as tf
import numpy as np
import time

# NAME = f"cnn-64x2-0x0D-layers-0-dropout-0-pool-relu-{time.time()}"
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"logs/{NAME}")
data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

x_train = np.concatenate((np.array(x_train), np.array(x_test)), axis=0)
y_train = np.concatenate((np.array(y_train), np.array(y_test)), axis=0)
x_train = tf.keras.utils.normalize(x_train)
X = np.reshape(x_train, (-1, 28, 28, 1))
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(30, (5,5), input_shape=X.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(15, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Conv2D(14, (2, 2)))
# model.add(tf.keras.layers.Activation('relu'))
# # model.add(tf.keras.layers.MaxPool2D((2, 2)))
#
# model.add(tf.keras.layers.Conv2D(7, (2, 2)))
# model.add(tf.keras.layers.Activation('relu'))
# # model.add(tf.keras.layers.MaxPool2D((1, 1)))

model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y_train, validation_split=0.3, epochs=10,batch_size=200,verbose=2)
model.evaluate(X, y_train,batch_size=200,verbose=0)
model.save('neuralnet12.model')