import keras
from keras.layers import *
from keras import backend as K
from keras import Model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
from PIL import Image
class res2net():
    def __init__(self, inputs, expansion):
        self.inputs = inputs
        self.expansion = expansion
        self.shape = self.inputs.get_shape().as_list()
        self.filters = self.shape[3] // self.expansion # 计算出滤波器个数

    def Conv3x3(self, inputs):
        print('filters!', self.filters)
        # 返回一个3x3卷积核的卷积
        conv3x3 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(inputs)
        batch = BatchNormalization()(conv3x3)
        return Activation('relu')(batch)

    def Add3x3(self, inputs_1, inputs_2):

        add_1 = Add()([inputs_1, inputs_2])
        return self.Conv3x3(add_1)

    def split(self, inputs):
        # 进行通道分离
        return Lambda(lambda x: tf.split(x, self.expansion, axis=3))(inputs)

    def res2netblock(self):
        split_layer = self.split(self.inputs)
        y1 = split_layer[0]
        print(y1.shape)
        print(split_layer[1].shape)
        y2 = self.Conv3x3(split_layer[1])
        print(y2.shape)
        y3 = self.Add3x3(y2, split_layer[2])
        y4 = self.Add3x3(y3, split_layer[3])
        return Concatenate(axis=-1)([y1, y2, y3, y4])
    
# stego_gen = keras.preprocessing.image.ImageDataGenerator()
# stego_data = stego_gen.flow_from_directory('./train/stego/', batch_size=2,
#                                            target_size=(224, 224))
#
# cover_gen = keras.preprocessing.image.ImageDataGenerator()
# cover_data = stego_gen.flow_from_directory('./train/cover/', batch_size=2,
#                                            target_size=(224, 224))



cover_label = np.array([[1, 0]])
cover_image = Image.open(r'D:\PycharmProject\res2net\train\cover\000001.jpg')
cover_image = cover_image.resize((224, 224))
cover_image = np.expand_dims(cover_image, axis=0)

inputs = Input((224, 224, 3))

x = Conv2D(filters=128, kernel_size=3, padding='SAME')(inputs)
print(x.shape)
# split = Lambda(lambda x:tf.split(x, 4, axis=3))(x)

res2net_layer = res2net(x, 4)
res2net_out = res2net_layer.res2netblock()
flatten = Flatten()(res2net_out)
out = Dense(2)(flatten)

model = Model(inputs, out)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', write_graph=True)

model.summary()
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MAE, metrics=['accuracy'])
model.fit(cover_image, cover_label, epochs=1, verbose=2, callbacks=[tbCallBack])
plot_model(model, 'res2netblock.png', show_shapes=True)