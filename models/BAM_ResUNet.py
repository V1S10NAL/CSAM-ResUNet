import tensorflow as tf
from tensorflow.keras import layers
import tools
from tensorflow.keras.metrics import CosineSimilarity
from models import CSAM_ResUNet


def bam_block(input_tensor, reduction_ratio=8, dilation_rate=4):
    """Bottleneck Attention Module (BAM)"""
    num_channels = input_tensor.shape[-1]

    # 通道注意力分支
    channel_avg = layers.GlobalAveragePooling1D()(input_tensor)  # (B, C)
    channel_avg = layers.Reshape((1, num_channels))(channel_avg)  # (B, 1, C)
    hidden_units = num_channels // reduction_ratio
    # MLP with Conv1D (equivalent to Dense layers)
    channel_avg = layers.Conv1D(hidden_units, kernel_size=1, activation='relu')(channel_avg)
    channel_attention = layers.Conv1D(num_channels, kernel_size=1, activation='sigmoid')(channel_avg)  # (B, 1, C)

    # 空间注意力分支
    spatial_avg = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)  # (B, L, 1)
    spatial_max = tf.reduce_max(input_tensor, axis=-1, keepdims=True)  # (B, L, 1)
    spatial_concat = layers.concatenate([spatial_avg, spatial_max], axis=-1)  # (B, L, 2)
    # 多层卷积处理
    spatial_conv = layers.Conv1D(64, kernel_size=3, padding='same', dilation_rate=dilation_rate, activation='relu')(
        spatial_concat)
    spatial_conv = layers.Conv1D(32, kernel_size=3, padding='same', dilation_rate=dilation_rate, activation='relu')(
        spatial_conv)
    spatial_conv = layers.Conv1D(32, kernel_size=3, padding='same', dilation_rate=dilation_rate, activation='relu')(
        spatial_conv)
    spatial_attention = layers.Conv1D(1, kernel_size=1, padding='same', activation='sigmoid')(spatial_conv)  # (B, L, 1)

    # 合并注意力（广播相加）
    attention = layers.Add()([channel_attention, spatial_attention])  # 自动广播到(B, L, C)
    attention = tf.sigmoid(attention)

    # 应用注意力权重
    output = layers.Multiply()([input_tensor, attention])
    return output

def residual_block(input_tensor, num_filters):
    """定义一个残差块，加入 BAM 块"""
    # 主路径
    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 加入 SE 块
    x = bam_block(x, num_filters)

    # 跳跃连接
    if input_tensor.shape[-1] != num_filters:
        shortcut = layers.Conv1D(num_filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    # 合并主路径和跳跃连接
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """定义一个编码器块（残差块 + 最大池化层）"""
    x = residual_block(input_tensor, num_filters)
    p = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    """定义一个解码器块（上采样 + 跳跃连接 + 残差块）"""
    x = layers.Conv1DTranspose(num_filters, kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])  # 跳跃连接
    x = residual_block(x, num_filters)
    return x

def BAM_ResUNet(num_features, num_outputs, args):
    """构建 ResUNet 模型，加入 BAM 块"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64)  # 第一阶段
    x2, p2 = encoder_block(p1, 128)     # 第二阶段
    x3, p3 = encoder_block(p2, 256)     # 第三阶段
    x4, p4 = encoder_block(p3, 512)     # 第四阶段

    # 桥接层
    b1 = residual_block(p4, 1024)

    # 解码器
    d1 = decoder_block(b1, x4, 512)  # 第一阶段
    d2 = decoder_block(d1, x3, 256)  # 第二阶段
    d3 = decoder_block(d2, x2, 128)  # 第三阶段
    d4 = decoder_block(d3, x1, 64)   # 第四阶段

    # 输出层（1个长度为 1024 的一维数据）

    outputs = layers.Conv1D(num_outputs, kernel_size=1, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-5)
                            )(d4)

    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 使用学习率调度器
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.9772,  # 0.955，100个epoch内lr下降到 1%
        staircase=False
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # 编译模型（回归任务，使用均方误差作为损失函数）
    model.compile(optimizer=adam_optimizer,
                  loss='mean_squared_error',
                  metrics=['mae', 'mape', 'mse',
                           CosineSimilarity(axis=-1),
                           tools.LogCoshError()])

    model.summary()

    return model

def BAM_ResUNet_classification(num_features, num_outputs, args):
    """构建 ResUNet 模型，加入 BAM块， 用于classification"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64)  # 第一阶段
    x2, p2 = encoder_block(p1, 128)     # 第二阶段
    x3, p3 = encoder_block(p2, 256)     # 第三阶段
    x4, p4 = encoder_block(p3, 512)     # 第四阶段

    # 桥接层
    b1 = residual_block(p4, 1024)


    # 回归分支----------------------------------------------------------------------
    # 解码器
    d1 = decoder_block(b1, x4, 512)  # 第一阶段
    d2 = decoder_block(d1, x3, 256)  # 第二阶段
    d3 = decoder_block(d2, x2, 128)  # 第三阶段
    d4 = decoder_block(d3, x1, 64)   # 第四阶段

    # 输出层（2个长度为 1024 的一维数据）
    outputs1 = layers.Conv1D(num_outputs, kernel_size=1, activation='relu',name='spectrum_A')(d4)
    outputs2 = layers.Conv1D(num_outputs, kernel_size=1, activation='relu',name='spectrum_B')(d4)

    # 分类分支----------------------------------------------------------------------
    # 全局池化 + 全连接
    c = bam_block(b1)
    c = layers.Conv1D(1024, kernel_size=1, activation='relu',name='classification_branch')(c)
    c_avg = layers.GlobalAveragePooling1D()(c)
    c_max = layers.GlobalMaxPooling1D()(c)
    c = layers.Concatenate()([c_avg, c_max])

    # 全连接层
    c = layers.Dense(512, activation='relu')(c)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(0.5)(c)
    c = layers.Dense(256, activation='relu')(c)
    c = layers.Dropout(0.5)(c)
    c = layers.Dense(args.num_substances, activation='sigmoid', name='classification')(c)
    # 创建模型----------------------------------------------------------------------

    model = tf.keras.Model(inputs=inputs,
                           outputs=[outputs1,outputs2,c]
                           )

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.977,
        # 0.955 : 100个epoch后lr下降到初始1%，lr drops to the initial 1% after 100 epochs;;
        # 0.977 :100个epoch后lr下降到初始10%，lr drops to the initial 10% after 100 epochs;
        # 0.9885 : 200个epoch后lr下降到初始10%，lr drops to the initial 10% after 200 epochs;；
        staircase=False
    )

    loss_instance = CSAM_ResUNet.DynamicHybridPhysicsLoss(
        mse_weight=1.0,
        sparse_strength=0.005,
        smooth_strength=0.005,
        ramp_epochs=100,
        gamma=100
    )

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    # 定义损失函数

    # 编译模型
    model.compile(optimizer=adam_optimizer,
                  loss={
                      'spectrum_A': loss_instance,                  # 回归损失1，Regression loss 1
                      'spectrum_B': loss_instance,                  # 回归损失2，Regression loss 2
                      'classification': 'binary_crossentropy'  # 分类损失，Classification loss
                  },
                  loss_weights={
                      'spectrum_A': 1.0,
                      'spectrum_B': 1.0,
                      'classification': 0.5
                  },
                  metrics={
                      'spectrum_A': ['mse', 'mae'],  # 回归分支1的指标
                      'spectrum_B': ['mse', 'mae'],  # 回归分支2的指标
                      'classification': [
                          'binary_accuracy',    #accuracy
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                      ]
                  }
    )

    model.summary()

    return model