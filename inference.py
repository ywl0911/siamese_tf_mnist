import tensorflow as tf


class siamese:

    # Create model
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])

        with tf.variable_scope("siamese") as scope:
            # self.o1 = self.network(self.x1)
            # scope.reuse_variables()
            # # 共用siamese下面所有的变量
            # self.o2 = self.network(self.x2)

            self.o1 = self.CNN_network(self.x1)
            scope.reuse_variables()
            # 共用siamese下面所有的变量
            self.o2 = self.CNN_network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        print(fc1.name)
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")  # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")

        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")  # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Nyi_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def CNN_network(self, x):
        def weight_variable(shape):
            return tf.get_variable(name='W', initializer=tf.truncated_normal(shape, stddev=0.1))

        def bias_variable(shape):
            return tf.get_variable(name='b', initializer=tf.truncated_normal(shape, stddev=0.1))

        with tf.variable_scope("conv1"):
            W_conv1_fliter = weight_variable([5, 5, 1, 32])
            # 5,5为卷积核patch的尺寸，1为输入的channel数量，32为输出的channel数量
            # [filter_height, filter_width, in_channels, out_channels]这样的shape，
            # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] # 第三维为in_channels，就是参数x_image的第四维  第四维为卷积核的个数
            b_conv1 = bias_variable([32])

            x_image = tf.reshape(x, [-1, 28, 28])
            # [batch, in_height, in_width, in_channels]最后一维为通道数量，可以理解为RGB，黑白图只有一个通道
            x_image_input = tf.expand_dims(x_image, -1)  # -1参数表示第几维，-1表示扩展最后一维
            # 给x_image加一维，从[-1, 28, 28]变为[-1, 28, 28,1]，最后一维为通道数量，等价于x_image_input = tf.reshape(x, [-1, 28, 28,1])

            h_conv1_z = tf.nn.conv2d(x_image_input, W_conv1_fliter, strides=[1, 1, 1, 1], padding='SAME')
            # # 第三个参数strides：卷积时在图像X每一维的步长，这是一个一维的向量，长度4
            # string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。”SAME”是考虑边界，不足的时候用0去填充周围，”VALID”则不考虑
            h_conv1_a = tf.nn.tanh(h_conv1_z + b_conv1)

            h_pool1 = tf.nn.max_pool(h_conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，维度为[batch, height, width, channels]
            # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
            # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
            # 第四个参数padding：和卷积类似，可以取    # 'VALID'    # 或者    # 'SAME'
            # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]  这种形式

        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])

            # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)

            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('flat'):
            # cnn之后的全连接层
            W_fc1 = weight_variable([7 * 7 * 64, 2])
            b_fc1 = bias_variable([2])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # h_pool2的维度为[batchsize，每张图片总共的像素点的个数]，经过两次卷积和两次pooling后，图片的变成了7×7，图片的通道变成了64
            h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
            #  经过全连接层后h_fc1的维度为 [batchsize，1024]
            # h_fc2=tf.sigmoid
        return h_fc1
