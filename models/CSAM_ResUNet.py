import math
from tensorflow.keras import layers
from keras import backend as K
import tools
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.metrics import CosineSimilarity


class DynamicHybridPhysicsLoss(tf.keras.losses.Loss):
    """
    动态混合物理损失函数，结合加权MSE、稀疏约束和平滑约束的
    Dynamic hybrid physical loss function combining weighted MSE, sparse constraint and smooth constraint
    :param mse_weight: base weighted MSE loss weight #基本加权的MSE损失权值
    :param sparse_strength: initial strength of sparse constraints #稀疏约束的初始强度
    :param smooth_strength: smooth constraint Initial strength #平滑约束初始强度
    :param ramp_epochs: number of constraint strength growth cycles #约束强度增长周期数
    :param gamma: constraint strength growth coefficient #约束强度增长系数
    """
    def __init__(self,
                 mse_weight=1.0,
                 sparse_strength=0.01,
                 smooth_strength=0.01,
                 ramp_epochs=100,
                 gamma=100.0,
                 ):
        super().__init__()

        self.mse_weight = tf.cast(mse_weight, tf.float32)

        # 保存初始强度值（用于指数增长计算）
        # Save the initial strength value (for exponential growth calculation)
        self.initial_sparse = tf.Variable(
            float(sparse_strength),
            trainable=False,
            dtype=tf.float32
        )
        self.initial_smooth = tf.Variable(
            float(smooth_strength),
            trainable=False,
            dtype=tf.float32
        )

        # 当前强度变量（将被动态更新）
        # Current strength variable (will be updated dynamically)
        self.sparse_strength = tf.Variable(
            float(sparse_strength),
            trainable=False,
            dtype=tf.float32
        )
        self.smooth_strength = tf.Variable(
            float(smooth_strength),
            trainable=False,
            dtype=tf.float32
        )

        self.ramp_epochs = tf.cast(ramp_epochs, tf.float32)
        self.gamma = tf.cast(gamma, tf.float32)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.float32)

    def get_config(self):
        return {
            "mse_weight": self.mse_weight.numpy(),
            "sparse_strength": self.initial_sparse.numpy(),
            "smooth_strength": self.initial_smooth.numpy(),
            "ramp_epochs": self.ramp_epochs.numpy(),
            "gamma": self.gamma.numpy(),
        }

    def update_strength(self, epoch):
        """
        "Update the weight coefficient according to the epoch (growing exponentially)"
        根据 epoch 更新权重系数（指数增长）
        """
        ratio = tf.minimum(
            tf.cast(epoch, tf.float32) / self.ramp_epochs,
            1.0
        )
        # 计算指数增长后的新强度
        # Calculate the new intensity after exponential growth
        new_sparse = self.initial_sparse * (self.gamma ** ratio)
        new_smooth = self.initial_smooth * (self.gamma ** ratio)

        # 更新强度变量
        self.sparse_strength.assign(new_sparse)
        self.smooth_strength.assign(new_smooth)
        self.current_epoch.assign(tf.cast(epoch, tf.float32))

    def call(self, y_true, y_pred, sample_weight=None):
        # y_true: 原始噪声光谱（非纯净光谱）. Original noise spectrum (non-pure spectrum)
        # y_pred: 模型预测的重构光谱. The reconstructed spectrum predicted by the model
        # sample_weight: 遮蔽区域的二进制掩码（仅计算遮蔽区域的损失）Binary mask of the masked area (only calculate the loss of the masked area)

        ## 1. 加权MSE损失
        ## 1. weighted MSE loss
        sample_mask = tf.cast(sample_weight, y_pred.dtype) if sample_weight is not None else tf.ones_like(y_true)
        mse_loss = self.mse_weight * tf.reduce_mean(sample_mask * tf.square(y_true - y_pred))

        ## 2. 非遮蔽区域稀疏约束（y_true低值区域）
        ## 2. Sparse constraint in unmasked areas (low-value areas of y_true)
        threshold = 1e-4
        sparse_mask = tf.cast(tf.less_equal(y_true, threshold), y_pred.dtype)
        sparse_loss = tf.reduce_mean(sparse_mask * tf.square(y_pred))

        ## 3. 平滑约束
        ## 3. Smooth Constraints
        def compute_second_derivative(tensor):
            d2 = tensor[:, 2:] - 2 * tensor[:, 1:-1] + tensor[:, :-2]
            d2 = tf.pad(d2, [[0, 0], [1, 1], [0, 0]], 'SYMMETRIC')
            return d2

        def compute_first_derivative(tensor):
            d1 = tensor[:, 1:] - tensor[:, :-1]
            d1 = tf.pad(d1, [[0, 0], [1, 0], [0, 0]], 'SYMMETRIC')
            return d1

        d2y_pred = compute_second_derivative(y_pred)
        smooth_loss = tf.reduce_mean(tf.square(d2y_pred))

        ## Total loss
        total_loss = (
            mse_loss +
            self.sparse_strength * sparse_loss +
            self.smooth_strength * smooth_loss
        )
        return total_loss

class StopAfterEpochsDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    带停止机制的指数衰减策略
    Exponential attenuation strategy with stopping mechanism
    :param initial_lr: Initial learning rate
    :param decay_steps: Decaying step size
    :param decay_rate: Attenuation coefficient
    :param max_epochs: Maximum number of training epochs
    """
    def __init__(self, initial_lr, decay_steps, decay_rate, max_epochs, batch_size, num_spectra):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
        self.max_steps = max_epochs * (num_spectra*0.7 // batch_size)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        max_steps = tf.cast(self.max_steps, tf.float32)
        return tf.cond(
            step < max_steps,
            lambda: self.decay_schedule(step),
            lambda: self.decay_schedule(max_steps)  # 超过max_steps后固定学习率. The learning rate remains unchanged after exceeding max_steps
        )

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "decay_steps": self.decay_schedule.decay_steps,
            "decay_rate": self.decay_schedule.decay_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "num_spectra": self.num_spectra
        }

def channel_attention(inputs):
    """
    参考双卷积eca优化的通道注意力
    Refer to the channel attention optimized by double convolution eca
    https://arxiv.org/abs/1910.03151
    """

    avg_pool = layers.GlobalAveragePooling1D(keepdims=True)(inputs)
    max_pool = layers.GlobalMaxPooling1D(keepdims=True)(inputs)
    # 跨通道交互（1D卷积替代全连接）
    avg = layers.Conv1D(1, kernel_size=5, padding='same',
                       use_bias=False, activation='relu', kernel_initializer='he_normal')(avg_pool)
    avg = layers.Conv1D(1, kernel_size=3, padding='same',
                       use_bias=False, activation='sigmoid', kernel_initializer='he_normal')(avg)

    max = layers.Conv1D(1, kernel_size=5, padding='same',
                       use_bias=False, activation='relu', kernel_initializer='he_normal')(max_pool)
    max = layers.Conv1D(1, kernel_size=3, padding='same',
                        use_bias=False, activation='sigmoid', kernel_initializer='he_normal')(max)
    combined = layers.Add()([avg, max])

    x = layers.Activation('sigmoid')(combined)

    # _, steps, channels = inputs.shape
    # avg_pool = layers.GlobalAveragePooling1D(keepdims=True)(inputs)  # (batch, 1, channels)
    # max_pool = layers.GlobalMaxPooling1D(keepdims=True)(inputs)      # (batch, 1, channels)
    #
    # # # 转置为 (batch, channels, 1)
    # # avg_pool = layers.Permute((2, 1))(avg_pool)
    # # max_pool = layers.Permute((2, 1))(max_pool)
    #
    # # 动态计算 kernel_size（参考 ECA 论文）
    # kernel_size = int(math.floor((channels + 1) / 2))
    # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    #
    # # 跨通道交互
    # avg = layers.Conv1D(1, kernel_size=kernel_size, padding='same',
    #                     use_bias=False, activation='relu', kernel_initializer='he_normal')(avg_pool)
    # avg = layers.Conv1D(1, kernel_size=3, padding='same',
    #                     use_bias=False, activation='sigmoid', kernel_initializer='he_normal')(avg)
    #
    # max_out = layers.Conv1D(1, kernel_size=kernel_size, padding='same',
    #                         use_bias=False, activation='relu', kernel_initializer='he_normal')(max_pool)
    # max_out = layers.Conv1D(1, kernel_size=3, padding='same',
    #                         use_bias=False, activation='sigmoid', kernel_initializer='he_normal')(max_out)
    #
    # combined = layers.Add()([avg, max_out])  # (batch, channels, 1)
    # x = layers.Activation('sigmoid')(combined)
    # # x = layers.Permute((2, 1))(x)  # 恢复为 (batch, 1, channels)


    return x

def spatial_attention(inputs, reduction_ratio=8):
    """
    参考BAM优化的瓶颈层空间注意力
    Refer to the bottleneck layer spatial attention optimized by BAM
    https://arxiv.org/abs/1807.06514
    """
    _, steps, channels = inputs.shape

    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # 下采样层
    x = layers.Conv1D(filters=channels // reduction_ratio, kernel_size=3, padding='same')(concat)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=channels // reduction_ratio, kernel_size=3,
                       padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 上采样恢复分辨率
    x = layers.Conv1D(1, kernel_size=1, padding='same')(x)
    x = layers.Activation('sigmoid')(x)  # 输出空间注意力掩码
    return x

def CSAM_block(inputs):
    """
    CSAM模块：通道注意力 + 空间注意力
    CSAM Module: Channel Attention + Spatial Attention
    """

    # Channel Attention
    ca = channel_attention(inputs)
    x = layers.multiply([inputs, ca])

    # Spatial Attention
    sa = spatial_attention(x)
    x = layers.multiply([x, sa])

    return x

def residual_block(input_tensor, num_filters):
    """加入 CSAM_block 块的残差块"""
    # 主路径
    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(num_filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 加入 CSAM 模块
    x = CSAM_block(x)

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

def encoder_block(input_tensor, num_filters, num_blocks):
    """编码器块（残差块 + 最大池化层）"""
    x = residual_block(input_tensor, num_filters)
    for _ in range(1, num_blocks):
        x = residual_block(x, num_filters)
    p = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    """解码器块（上采样 + 跳跃连接 + 残差块）"""
    x = layers.Conv1DTranspose(num_filters, kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])  # 跳跃连接
    x = residual_block(x, num_filters)
    return x

def CSAMResUNet_reconstruction(num_features, num_outputs, args):
    """ResUNet 模型，加入 CSAM 块, 用于reconstruction"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64,1)  # 第一阶段
    x2, p2 = encoder_block(p1, 128,1)     # 第二阶段
    x3, p3 = encoder_block(p2, 256,1)     # 第三阶段
    x4, p4 = encoder_block(p3, 512,1)     # 第四阶段

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


    # 配置学习率调度
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.9772,
        # 0.955 : 100个epoch后lr下降到初始1%，lr drops to the initial 1% after 100 epochs;
        # 0.9772 :100个epoch后lr下降到初始10%，lr drops to the initial 10% after 100 epochs;
        # 0.9847: 150个epoch后lr下降到初始10%，lr drops to the initial 10% after 150 epochs;
        # 0.9885 : 200个epoch后lr下降到初始10%，lr drops to the initial 10% after 200 epochs;
        staircase=False
    )

    # 创建损失实例
    # Create loss instances
    loss_instance = DynamicHybridPhysicsLoss(
        mse_weight=1.0,
        sparse_strength=0.005,
        smooth_strength=0.005,
        ramp_epochs=100,
        gamma=100,
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # 编译模型（回归任务，使用 自定义损失函数 作为损失函数）
    # Compilation Model (Regression task, using a custom loss function as the loss function)
    model.compile(optimizer=adam_optimizer,
                  loss=loss_instance,
                  #loss='mean_squared_error',
                  metrics=['mae', 'mape', 'mse',
                           CosineSimilarity(axis=-1),
                           tools.LogCoshError()])

    model.summary()

    return model

def CSAMResUNet_classification(num_features, num_outputs, args):
    """构建 ResUNet 模型，加入 CSAM 块， 用于classification"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64,1)  # 第一阶段
    x2, p2 = encoder_block(p1, 128,1)     # 第二阶段
    x3, p3 = encoder_block(p2, 256,1)     # 第三阶段
    x4, p4 = encoder_block(p3, 512,1)     # 第四阶段

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
    c = CSAM_block(b1)
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

    loss_instance = DynamicHybridPhysicsLoss(
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

def CSAMResUNet_classification2(num_features, num_outputs, args):
    """构建 ResUNet 模型，加入 CSAM 块， only classification，without unmixing"""
    inputs = layers.Input(shape=(num_features, 1))

    # 编码器
    x1, p1 = encoder_block(inputs, 64, 1)  # 第一阶段
    x2, p2 = encoder_block(p1, 128, 1)  # 第二阶段
    x3, p3 = encoder_block(p2, 256, 1)  # 第三阶段
    x4, p4 = encoder_block(p3, 512, 1)  # 第四阶段

    # 桥接层
    b1 = residual_block(p4, 1024)

    # 分类分支----------------------------------------------------------------------
    c = CSAM_block(b1)
    c = layers.Conv1D(1024, kernel_size=1, activation='relu', name='classification_branch')(c)
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
    model = tf.keras.Model(inputs=inputs, outputs=[c])  # 只保留分类输出

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.977,
        staircase=False
    )

    loss_instance = DynamicHybridPhysicsLoss(
        mse_weight=1.0,
        sparse_strength=0.005,
        smooth_strength=0.005,
        ramp_epochs=100,
        gamma=100
    )

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # 编译模型
    model.compile(optimizer=adam_optimizer,
                  loss={'classification': 'binary_crossentropy'},  # 只保留分类损失
                  loss_weights={'classification': 0.5},  # 调整权重
                  metrics={
                      'classification': [
                          'binary_accuracy',  # accuracy
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                      ]
                  }
                  )

    model.summary()

    return model
