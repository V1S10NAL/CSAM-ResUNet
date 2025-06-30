"""
ResUNet 结合自注意力机制（Self-Attention）模块
"""
import tensorflow as tf
from tensorflow.keras import layers


# 自定义自注意力层
class SelfAttention(layers.Layer):
    def __init__(self, num_heads=1, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.d_model = input_shape[-1]  # 获取输入通道数
        # 创建Q、K、V的权重矩阵
        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)  # 最终输出投影

    def call(self, inputs):
        # 生成Q、K、V
        q = self.wq(inputs)  # (batch_size, seq_len, d_model)
        k = self.wk(inputs)  # (batch_size, seq_len, d_model)
        v = self.wv(inputs)  # (batch_size, seq_len, d_model)

        # 计算注意力分数
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, seq_len, seq_len)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        output = self.dense(output)

        output += inputs
        return output

    # 新增的序列化方法
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
        })
        return config

def residual_block(input_tensor, num_filters):
    """定义一个残差块"""

    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if input_tensor.shape[-1] != num_filters:
        shortcut = layers.Conv1D(num_filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    x = SelfAttention()(x)
    return x

def encoder_block(input_tensor, num_filters):
    """定义一个编码器块（残差块 + 最大池化层）"""
    x = residual_block(input_tensor, num_filters)
    p = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    """定义一个解码器块（上采样 + 跳跃连接 + 残差块）"""
    x = layers.Conv1DTranspose(num_filters, kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = residual_block(x, num_filters)
    return x

def saResUNet(num_features, num_outputs, args):
    """构建 Self-Attention ResUNet 模型"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64)  # 第一阶段
    x2, p2 = encoder_block(p1, 128)     # 第二阶段
    x3, p3 = encoder_block(p2, 256)     # 第三阶段
    x4, p4 = encoder_block(p3, 512)     # 第四阶段

    # 桥接层
    b1 = residual_block(p4, 1024)
    b1 = SelfAttention()(b1)
    # 解码器
    d1 = decoder_block(b1, x4, 512)  # 第一阶段
    d2 = decoder_block(d1, x3, 256)  # 第二阶段
    d3 = decoder_block(d2, x2, 128)  # 第三阶段
    d4 = decoder_block(d3, x1, 64)   # 第四阶段


    outputs = layers.Conv1D(num_outputs, kernel_size=1, activation='relu')(d4)

    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 使用学习率调度器
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.9885,  # 0.955，100个epoch内lr下降到 1%
        staircase=False
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=adam_optimizer,
                  loss='mean_squared_error',
                  metrics=['mae', 'mape', 'mse'])

    model.summary()

    return model
