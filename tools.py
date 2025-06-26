from itertools import cycle
from tensorflow.keras import layers
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_curve, auc, confusion_matrix, \
    classification_report
import scipy.io
import numpy as np
import seaborn as sns
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
from scipy import interp
from sklearn.metrics import accuracy_score
from itertools import combinations
import os
import cv2
from models import ResUNet, CBAM_ResUNet, ECA_ResUNet, SA_ResUNet, SE_ResUNet, BAM_ResUNet, CSAM_ResUNet


# 定义保存预测图像的回调类（添加到模型训练部分之前）
class SaveAndVideoCallback(tf.keras.callbacks.Callback):
    """保存每个epoch的预测图并生成视频（可选开关） / Save prediction images per epoch and generate video (optional)"""

    def __init__(self, X_data, wavenumber, output_dir, generate_video=True, fps=15, sample_idx=0):
        """
        Args:
            X_data (np.ndarray): 输入数据 / Input data
            wavenumber (np.ndarray): 拉曼位移波长数组 / Raman shift wavenumber array
            output_dir (str): 输出目录路径 / Output directory path
            generate_video (bool): 是否生成图像和视频，默认True / Whether to generate images and video, default True
            fps (int): 视频帧率，默认15 / Video frames per second, default 15
            sample_idx (int): 选择可视化的样本索引，默认0 / Sample index to visualize, default 0
        """
        super().__init__()
        self.generate_video = generate_video
        if not self.generate_video:
            return  # 不初始化相关参数

        self.X_data = X_data
        self.wavenumber = wavenumber
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames and video")
        os.makedirs(self.frame_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not self.generate_video:
            return
        # 原图像生成逻辑
        y_pred = self.model.predict(self.X_data, verbose=0).reshape(-1)
        plt.figure(figsize=(16, 9))
        plt.plot(self.wavenumber, y_pred, linewidth=0.5)
        plt.title(f"Epoch {epoch + 1}")
        plt.xlabel('Raman shift')
        plt.ylabel('Intensity')
        frame_path = os.path.join(self.frame_dir, f"epoch_{epoch + 1:04d}.png")
        plt.savefig(frame_path, bbox_inches="tight", dpi=600)
        plt.close()


    def on_train_end(self, logs=None):
        if not self.generate_video:
            return
        # 原视频生成逻辑
        images = sorted(
            [f for f in os.listdir(self.frame_dir) if f.endswith(".png")],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        if not images:
            print("[Callback] video error")
            return
        first_frame = cv2.imread(os.path.join(self.frame_dir, images[0]))
        height, width, _ = first_frame.shape
        video_path = os.path.join(self.output_dir, "prediction_evolution.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        for img in images:
            video.write(cv2.imread(os.path.join(self.frame_dir, img)))
        video.release()
        print(f"video at: {video_path}")

# 自定义 LogCosh 指标
class LogCoshError(tf.keras.metrics.Metric):
    """自定义LogCosh指标 / Custom LogCosh Metric"""
    def __init__(self, name="log_cosh_error", **kwargs):
        """
        Args:
            name (str): 指标名称，默认"log_cosh_error" / Metric name, default "log_cosh_error"
            **kwargs: 其他父类参数 / Other parent class parameters
        """
        super(LogCoshError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = y_pred - y_true
        log_cosh_val = tf.math.log((tf.exp(error) + tf.exp(-error)) / 2)
        loss = tf.reduce_sum(log_cosh_val)
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=log_cosh_val.dtype)
        self.total.assign_add(loss)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / self.count

def dict_to_xlsx(dict_data, output_file):
    """
    将一个字典转换为DataFrame，并将其保存为xlsx文件 / Convert a dictionary to DataFrame and save as xlsx file

    Args:
        dict_data (dict): 字典数据，键对应不同长度的数据 / Dictionary with keys mapping to variable-length data
        output_file (str): 输出xlsx文件路径 / Output xlsx file path
    """
    # 创建一个空的DataFrame
    df = pd.DataFrame.from_dict(dict_data, orient='index').transpose()

    # 保存DataFrame为xlsx文件
    df.to_excel(output_file, index=False)

def plot_roc_curve(y_true, y_pred, class_names, output_image_path):
    """
    绘制多类别ROC曲线 / Plot multi-class ROC curve

    Args:
        y_true (np.ndarray): 真实标签矩阵 / Ground truth label matrix
        y_pred (np.ndarray): 预测概率矩阵 / Predicted probability matrix
        class_names (list): 类别名称列表 / List of class names
        output_image_path (str): 输出图像路径 / Output image path
    """
    plt.figure(figsize=(8, 6))

    # 计算每个类别的ROC曲线
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

def plot_roc_curves2(y_test, y_score, classes_list,output_image_path):
    """
    绘制多类别ROC曲线（支持微平均和宏平均） / Plot multi-class ROC curves (with micro/macro averaging)

    Args:
        y_test (np.ndarray): 测试集真实标签 / Ground truth labels of test set
        y_score (np.ndarray): 预测概率分数 / Predicted probability scores
        classes_list (list): 类别列表 / List of classes
        output_image_path (str): 输出图像路径 / Output image path
    """
    # 计算每一类的ROC
    # y_test = label_binarize(y_test_tensor, classes=classes_list)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # 对于每个类别，计算其假正率（FPR）和真正率（TPR），并计算AUC。
    for i in range(len(classes_list)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes_list))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes_list)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes_list)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(6, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.9f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.9f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['cadetblue', 'skyblue', 'lightsteelblue', 'cornflowerblue', 'mediumslateblue', 'plum'])  # 曲线颜色
    for i, color in zip(range(len(classes_list)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of {classes_list[i]} (area = {roc_auc[i]:0.9f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.legend(fontsize='small')
    # 保存图像

    plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')  # 输出


    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()
    print(' ')
    # 保存矩阵为.txt文件，便于origin重绘
    # output = './result/classification_report_' + current_time + '.txt'
    # with open(output, 'w', encoding='utf-8') as file:
    #     file.write(report)
    # print("\n", "classification_report", "\n")
    # print(report)
    return fpr, tpr, roc_auc

def plot_multilabel_confusion_matrix(y_true, y_pred, class_names, output_image_path, threshold=0.5, normalize=False, figsize=(12, 5)):
    """
    多标签分类混淆矩阵可视化 / Visualization of multi-label confusion matrix

    Args:
        y_true (np.ndarray): 真实标签矩阵 / Ground truth label matrix
        y_pred (np.ndarray): 预测概率矩阵 / Predicted probability matrix
        class_names (list): 类别名称列表 / List of class names
        output_image_path (str): 输出图像路径 / Output image path
        threshold (float): 分类阈值，默认0.5 / Classification threshold, default 0.5
        normalize (bool): 是否显示百分比，默认False / Whether to normalize values, default False
        figsize (tuple): 图像尺寸，默认(12,5) / Figure size, default (12,5)
    """
    # 将概率转换为二进制预测
    y_pred_binary = (y_pred > threshold).astype(int)

    # 创建子图网格
    n_classes = len(class_names)
    n_cols = min(3, n_classes)  # 每行最多3个子图
    n_rows = (n_classes + n_cols - 1) // n_cols

    plt.figure(figsize=figsize)

    for i, name in enumerate(class_names):
        ax = plt.subplot(n_rows, n_cols, i + 1)

        # 计算混淆矩阵
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            vmax = 1.0
            fmt = ".2f"
            title_suffix = "(Normalized)"
        else:
            vmax = cm.max()
            fmt = "d"
            title_suffix = "(Counts)"

        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=['Neg', 'Pos'],
                    yticklabels=['Neg', 'Pos'],
                    vmin=0, vmax=vmax,
                    cbar=False, ax=ax)

        ax.set_title(f'{name} {title_suffix}\nTP={cm[1, 1]}, FP={cm[0, 1]}\nFN={cm[1, 0]}, TN={cm[0, 0]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

def plot_cm(labels, predictions, xticks, fig_path, report_path, target_names, normalize=False):
    """
    绘制混淆矩阵
    :param fig_path:
    :param labels: 真实标签
    :param predictions: 预测标签
    :param xticks: x轴，mp类型
    :param normalize:
    :return:
    """
    # 将独热编码转换为类别索引
    labels = np.argmax(labels, axis=1)
    predictions = np.argmax(predictions, axis=1)

    cm = tf.math.confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    # sns.heatmap(cm, annot=True, fmt="d")
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n混淆矩阵显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('\n混淆矩阵显示具体数字：')
        print(cm)

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    # plt.title('Confusion matrix ')
    # plt.xlabel('Predicted label')
    # plt.ylabel('Actual label')
    # plt.xticks(range(len(xticks)), labels=xticks, rotation = 30)
    # plt.yticks(range(len(xticks)), labels=xticks)
    # plt.tight_layout()
    # ===============================================
    # 设置图形参数
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.0)  # 调整字体大小

    # 绘制热图并显示数值
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar=True,
        annot_kws={"size": 8}  # 数值字体大小
    )

    # 设置坐标轴标签和标题
    ax.set(
        xlabel='Predicted Label',
        ylabel='True Label',
        title=f'Confusion Matrix'
    )
    plt.xticks(ticks=np.arange(len(xticks)) + 0.5, labels=xticks, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(xticks)) + 0.5, labels=xticks, rotation=0)
    # ===============================================


    # 调整布局并保存
    plt.tight_layout()

    # fig_name = './fig/cm_pic_' + current_time + '.png'
    plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')  # 输出
    plt.show(block=False)
    plt.pause(2)  # 2 秒后继续执行
    plt.close()

    report = classification_report(labels, predictions, digits=4,
                                   zero_division=0,
                                   output_dict=False,
                                   target_names=target_names)
    # 保存为.txt文件

    with open(report_path, 'w', encoding='utf-8') as file:
        file.write(report)
    print("\n", "一维向量单标签 Classification Report")
    print(report)

def save_dict_to_txt(data, file_name, indent=0):
    """
    递归地将字典保存到 txt 文件
    :param data: 要保存的字典
    :param file_name: 文件名
    :param indent: 缩进级别（用于格式化）
    """
    with open(file_name, 'w') as f:
        def write_dict(d, indent_level):
            for key, value in d.items():
                # 如果键是内存地址，直接写入
                if isinstance(key, str) and key.startswith("<"):
                    f.write(" " * indent_level + f"内存地址: {key}\n")
                else:
                    f.write(" " * indent_level + f"{key}:\n")
                # 递归处理值
                if isinstance(value, dict):
                    write_dict(value, indent_level + 2)
                else:
                    f.write(" " * (indent_level + 2) + f"{value}\n")
                f.write("\n")

        write_dict(data, indent)

def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [MFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to MFLOPs
    return flops.total_float_ops / 1e6

def calculate_flops(model, batch_size=1, input_shape=None):
    """
    计算TensorFlow模型的GFLOPs（十亿浮点运算次数）。

    Args:
        model: 已构建的TensorFlow模型（如keras.Sequential或functional API模型）。
        batch_size: 输入张量的批量大小，默认为1。
        input_shape: 模型输入形状（不包含batch维度）。如果模型未构建，必须提供此参数。

    Returns:
        total_flops: 模型的总GFLOPs（十亿浮点运算次数）。
    """
    # 如果模型未构建，使用提供的input_shape进行构建
    if not model.built:
        if input_shape is None:
            raise ValueError("Model is not built and 'input_shape' not provided.")
        model.build(input_shape=(batch_size, *input_shape))

    # 创建输入TensorSpec（包含batch_size）
    input_shape_with_batch = [batch_size] + list(model.input.shape[1:])
    inputs = tf.TensorSpec(shape=input_shape_with_batch, dtype=model.input.dtype)

    # 创建模型的ConcreteFunction
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(inputs)

    # 配置分析选项（计算FLOPs）
    opts = option_builder.ProfileOptionBuilder.float_operation()
    # 可选：隐藏详细输出（设置为None则不显示）
    opts['output'] = 'none'

    # 执行分析
    flops_info = model_analyzer.profile(concrete_func.graph, options=opts)
    total_flops = flops_info.total_float_ops

    return total_flops /  1e9


def convert_to_1d(y_2d, N = 5):
    # 生成所有可能的双标签组合
    all_combinations = list(combinations(range(N), 2))
    num_classes = len(all_combinations)
    index_map = {combo: idx for idx, combo in enumerate(all_combinations)}

    # 初始化结果数组
    batch_size = y_2d.shape[0]
    y_1d_batch = np.zeros((batch_size, num_classes), dtype=int)

    for i in range(batch_size):
        # 提取单个样本
        sample = y_2d[i]
        # 找到为1的索引并排序
        active_indices = np.where(sample == 1)[0]
        sorted_indices = tuple(sorted(active_indices))

        # 映射到组合索引
        idx = index_map.get(sorted_indices, -1)
        if idx != -1:
            y_1d_batch[i, idx] = 1

    return y_1d_batch

def generate_gradcam_heatmap(model, input_sample, target_layer_suffix='classification_branch',
                             top_k=2, interpolation='bilinear'):
    """
    生成Grad-CAM热力图（支持选择前k个概率最高的类别）。

    参数:
    - model: 训练好的Keras模型
    - input_sample: 输入样本（形状为 (1, time_steps, 1)）
    - target_layer_suffix: 目标卷积层名称的后缀（如 'classification_branch'）
    - top_k: 返回前k个概率最高的类别对应的热力图（默认2）
    - interpolation: 插值方法（如 'bilinear'）

    返回:
    - heatmaps: 字典，键为类别索引，值为对应的热力图数组（形状与 input_sample 的 time_steps 一致）
    - top_classes: 概率最高的前k个类别索引列表
    """

    # 1. 寻找目标卷积层
    target_layer = None
    for layer in reversed(model.layers):
        if target_layer_suffix in layer.name and isinstance(layer, tf.keras.layers.Conv1D):
            target_layer = layer
            break
    if target_layer is None:
        raise ValueError(f"未找到包含 '{target_layer_suffix}' 的卷积层")
    conv_name = target_layer.name
    # print(f"Selected layer: {conv_name}")

    # 2. 获取分类输出层（假设名称为 'classification'）
    classification_output = model.get_layer('classification').output

    # 3. 构建梯度模型
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, classification_output]
    )

    # 4. 处理输入样本（添加 batch 维度）
    #input_var = tf.Variable(tf.expand_dims(input_sample, 0), dtype=tf.float32)  # 形状变为 (1, time_steps, 1)
    input_var = tf.Variable(input_sample, dtype=tf.float32)

    # 5. 获取所有类别预测概率
    with tf.GradientTape() as tape:
        tape.watch(input_var)
        conv_out, preds = grad_model(input_var)
        probs = tf.nn.sigmoid(preds)[0]  # 假设分类分支使用sigmoid激活（多标签）

    # 6. 获取前top_k个概率最高的类别索引
    top_classes = tf.argsort(probs, direction='DESCENDING')[:top_k].numpy()
    # print(f"Top {top_k} classes: {top_classes}")

    # 7. 计算每个选定类别的热力图
    heatmaps = {}
    for class_idx in top_classes:
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            conv_out, preds = grad_model(input_var)
            loss = preds[0][class_idx]  # 当前类别的预测概率

        grads = tape.gradient(loss, conv_out)
        conv_out_val = conv_out[0]  # 取出第一个样本的特征图 (time_steps, filters)
        grads_val = grads[0]  # (time_steps, filters)

        # 7.1 计算权重并生成热力图
        pooled_grads = tf.reduce_mean(grads_val, axis=0)  # (filters,)
        heatmap = tf.reduce_sum(conv_out_val * pooled_grads[tf.newaxis, :], axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)


        # 7.2 调整分辨率到原始输入长度
        # original_length = input_sample.shape[1] ##1024
        # if heatmap.shape[0] != original_length:
        #     heatmap = tf.image.resize(
        #         heatmap[tf.newaxis, :, tf.newaxis],
        #         [original_length, 1],
        #         method=interpolation
        #     )
        # heatmap =heatmap[:, 0, 0]
        #
        # heatmaps[class_idx] = heatmap.numpy()

        # === 调整热力图尺寸到原始输入长度 ===
        original_steps = input_var.shape[1]  # 输入样本的原始时间步数

        # 扩展维度适配resize操作（需要 [batch, height, width, channels] 格式）
        heatmap_expanded = tf.expand_dims(heatmap, axis=0)  # [1, conv_steps]
        heatmap_expanded = tf.expand_dims(heatmap_expanded, axis=-1)  # [1, conv_steps, 1]
        heatmap_expanded = tf.expand_dims(heatmap_expanded, axis=-1)  # [1, conv_steps, 1, 1]

        # 调整尺寸（提供二维尺寸参数）
        heatmap_resized = tf.image.resize(
            heatmap_expanded,
            size=[original_steps, 1],  # [new_height, new_width]
            method=interpolation
        )

        # 压缩回一维
        heatmap = tf.squeeze(heatmap_resized)  # 形状 (original_steps,)
        heatmaps[class_idx] = heatmap.numpy()

    return heatmaps, top_classes

def evaluate_multilabel_accuracy(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: list,
        var: str,
        var_to_labels: dict,
        type_mp,
        threshold: float = None,
        top_k: int = 2
) -> dict:
    """
    多标签分类评估函数（支持动态匹配物质组合）

    参数:
        y_true:        真实标签矩阵 (n_samples, n_classes)
        y_pred_proba:  预测概率矩阵 (n_samples, n_classes)
        class_names:   所有类别名称列表 (n_classes,)
        var:           当前物质组合标识（如 "PCPE"）
        var_to_labels: 物质组合映射字典（如 {"PCPE": ["PC", "PE"]}）
        threshold:     二值化阈值（None时使用top_k策略）
        top_k:         预测标签数量（当threshold=None时生效）

    返回:
        dict: 包含各指标的字典

    示例:
        var_to_labels = {"PCPE": ["PC", "PE"], "PCPP": ["PC", "PP"]}
        metrics = evaluate_multilabel_accuracy(y_true, y_pred, ["PC", "PE", "PP"], "PCPE", var_to_labels)
    """
    # ===== 1. 参数校验 =====
    # 验证var映射是否存在
    target_labels = var_to_labels.get(var)
    if not target_labels or len(target_labels) != 2:
        raise ValueError(f"无效的物质组合标识: {var} 或映射不完整")

    # 验证标签是否存在
    missing = [label for label in target_labels if label not in class_names]
    if missing:
        raise ValueError(f"标签 {missing} 不存在于 class_names 中")

    # ===== 2. 获取目标列索引 =====
    for index in range(len(type_mp)):
        if target_labels[0] == type_mp[index]:
            idx1 = index
        if target_labels[1] == type_mp[index]:
            idx2 = index
    print(idx1,idx2)

    # idx1 = class_names.index(target_labels[0])
    # idx2 = class_names.index(target_labels[1])

    # ===== 3. 生成预测结果 =====
    if threshold is not None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    else:
        # 仅针对目标标签计算Top-K（关键修改点）全局取前两个最高概率
        topk_indices = np.argsort(-y_pred_proba, axis=1)[:, :top_k]
        # 创建全零布尔掩码
        mask = np.zeros_like(y_pred_proba, dtype=bool)
        rows = np.arange(y_pred_proba.shape[0])[:, None]  # 生成行索引 [[0], [1], ..., [n-1]]

        # 将前top_k位置标记为True
        mask[rows, topk_indices] = True

        # 生成二进制标签
        y_pred_binary = mask.astype(int)
    # ===== 4. 提取目标标签数据 =====
    y_true_selected = y_true[:, [idx1, idx2]]
    y_pred_selected = y_pred_binary[:, [idx1, idx2]]

    # ===== 5. 计算指标 =====
    metrics = {}

    # 各标签准确率（按物质组合顺序）
    label_acc = {
        target_labels[0]: accuracy_score(y_true_selected[:, 0], y_pred_selected[:, 0]),
        target_labels[1]: accuracy_score(y_true_selected[:, 1], y_pred_selected[:, 1])
    }
    metrics["label_accuracy"] = label_acc

    # 子集准确率（双标签同时正确）
    metrics["subset_accuracy"] = accuracy_score(y_true_selected, y_pred_selected)

    # 样本平均准确率（按样本计算均值）
    sample_acc = (y_true_selected == y_pred_selected).mean(axis=1).mean()
    metrics["average_sample_accuracy"] = sample_acc

    return metrics

# 定义计算指标的函数
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    epsilon = 1e-8  # 防止除零的小值
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return r2, mse, rmse, mae, mape

# 从matlab将.mat文件转为numpy数组
def load_mat_to_np(path):
    """
    从matlab读取 .mat 文件中的所有变量，并将其转换为 NumPy 数组。
    :param path: .mat文件路径
    :return: numpy数组
    """
    # 读取 .mat 文件
    mat_data = scipy.io.loadmat(path)  #.mat里有多个矩阵时，返回一个字典
    # 存储变量名和对应的 NumPy 数组
    numpy_data = {}
    # 遍历键值对，提取变量并转换为 NumPy 数组
    for key, value in mat_data.items():
        if not key.startswith('__'):  # 忽略 MATLAB 的元数据（以 '__' 开头的键）
            if 'scipy.sparse' in str(type(value)):  # 检查是否为稀疏矩阵
                numpy_array = value.toarray()  # 将稀疏矩阵转换为稠密矩阵
            else:
                numpy_array = np.array(value)  # 将普通矩阵转换为 NumPy 数组
            numpy_data[key] = numpy_array  # 将变量名和 NumPy 数组存入字典

    return numpy_data

# 设置随机种子
def set_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    tf.random.set_seed(seed)  # TensorFlow 随机种子
    # 启用操作确定性（可选）
    tf.config.experimental.enable_op_determinism()

def snr(x, x_0):
    """
    计算原始光谱和去噪后光谱的信噪比（SNR）
    :param x: 去噪后光谱（干净光谱）
    :param x_0: 原始光谱（噪声光谱）
    :return: 每个样本的SNR值和平均SNR
    """
    n = x.shape[0]
    snr_all = np.zeros(n)

    for i in range(n):
        denoised_signal = x[i, :]  # 去噪后的信号
        original_signal = x_0[i, :]  # 原始信号

        # 正确计算噪声为原始信号减去去噪后的信号
        noise = original_signal - denoised_signal

        # 计算信号功率（去噪后信号的功率）和噪声功率
        signal_power = np.sum(denoised_signal ** 2)
        noise_power = np.sum(noise ** 2)

        # 处理噪声功率为零的情况
        if noise_power == 0:
            if signal_power == 0:
                snr_all[i] = np.nan  # 信号和噪声功率均为零时返回NaN
            else:
                snr_all[i] = np.inf  # 信号存在但噪声为零时返回无穷大
        else:
            # 计算SNR的dB值
            ratio = signal_power / noise_power
            if ratio == 0:
                snr_all[i] = -np.inf  # 信号功率为零且噪声存在时返回-无穷大
            else:
                snr_all[i] = 10 * np.log10(ratio)

    return snr_all, np.mean(snr_all)


def calculate_r2_score(y_pred, y_true):
    """
    计算两个形状为 [n, 1024] 的样本的 R² 值（按特征维度计算）

    参数:
        y_true (np.ndarray): 真实值，形状 [n, 1024]
        y_pred (np.ndarray): 预测值，形状 [n, 1024]

    返回:
        r2_scores (np.ndarray): 每个特征的 R² 值，形状 [1024,]
        mean_r2 (float): 所有特征的 R² 均值
    """
    # 计算残差平方和 (SS_res)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)

    # 计算总平方和 (SS_tot)
    y_true_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_true_mean) ** 2, axis=0)

    # 避免除以零（若 SS_tot=0，则 R²=1）
    valid_ss_tot = ss_tot != 0
    r2_scores = np.zeros_like(ss_res)
    r2_scores[valid_ss_tot] = 1 - ss_res[valid_ss_tot] / ss_tot[valid_ss_tot]
    r2_scores[~valid_ss_tot] = 1.0  # 若 SS_tot=0，预测完美

    # 计算均值
    mean_r2 = np.mean(r2_scores)

    return r2_scores, mean_r2

def peak_signal_noise_ratio(x,x_0):
    # 初始化 psnr 值
    n = x.shape[0]
    psnr_all = np.zeros(n)
    for i in range(n):
        max_val1 = np.max(x[i, :])
        max_val2 = np.max(x_0[i, :])
        max_val = max(max_val1,max_val2)
        mse = np.mean((x[i, :]-x_0[i, :])**2)
        psnr_all[i] = 10*np.log10((max_val**2/mse))
    return psnr_all, np.mean(psnr_all)

def cv(x):
    # 初始化 cv 值
    n = x.shape[0]
    cv_all = np.zeros(n)
    for i in range(n):
        mean_val = np.mean(x[i, :])
        # mean_val = np.max(x[i, :])
        std_val = np.std(x[i, :])

        cv_all[i] = std_val/ mean_val if mean_val != 0 else 0.0

    return cv_all, np.mean(cv_all)

def mrf(x):
    # 初始化 cv 值
    n = x.shape[0]
    mrf_all = np.zeros(n)
    for i in range(n):
        #mean_val = np.mean(x[i, :])
        max_val = np.max(x[i, :])
        std_val = np.std(x[i, :])

        mrf_all[i] = std_val/ max_val if max_val != 0 else 0.0

    return mrf_all, np.mean(mrf_all)

def calculate_mse(x,x_0):
    # 初始化 cv 值
    n = x.shape[0]
    mse_all = np.zeros(n)
    for i in range(n):
        mse_all[i] = np.mean((x[i,:] - x_0[i,:]) ** 2)
    return mse_all, np.mean(mse_all)

def ssim_1d(x, y, win_size=3, data_range=None, C1=None, C2=None):
    """
    计算一维光谱数据的 SSIM。

    参数:
        x (np.ndarray): 去噪后光谱，形状为 (n,1024)。
        y (np.ndarray): 带基线光谱，形状为 (n,1024)。
        win_size (int): 滑动窗口大小，必须是奇数。
        data_range (float): 数据的动态范围（最大值 - 最小值）。
        C1 (float): 常数 C1，默认为 (0.01 * data_range)^2。
        C2 (float): 常数 C2，默认为 (0.03 * data_range)^2。

    返回:
        ssim_all (np.ndarray): 整个光谱的 SSIM 值，形状为 (n,1)。
        ssim_mean (float): 所有 SSIM 值的平均值。
    """
    # 参数校验
    assert x.shape == y.shape, "输入光谱形状必须一致"
    assert win_size % 2 == 1, "窗口大小必须为奇数"

    n_samples, _ = x.shape

    # 自动计算动态范围（若未提供）
    if data_range is None:
        data_ranges = np.zeros(n_samples)
        for i in range(n_samples):
            max_val = max(x[i].max(), y[i].max())
            min_val = min(x[i].min(), y[i].min())
            data_ranges[i] = max_val - min_val
    else:
        data_ranges = np.full(n_samples, data_range)

    # 自动计算 C1 和 C2（若未提供）
    if C1 is None:
        C1 = (0.01 * data_ranges) ** 2
    else:
        C1 = np.full(n_samples, C1)

    if C2 is None:
        C2 = (0.03 * data_ranges) ** 2
    else:
        C2 = np.full(n_samples, C2)

    ssim_all = np.zeros(n_samples)

    # 遍历每个样本计算 SSIM
    for i in range(n_samples):
        xi = x[i]
        yi = y[i]
        dr_i = data_ranges[i]
        c1 = C1[i]
        c2 = C2[i]

        # 计算滑动窗口的均值
        mu_x = uniform_filter1d(xi, size=win_size, mode='nearest')
        mu_y = uniform_filter1d(yi, size=win_size, mode='nearest')

        # 计算方差和协方差
        sigma_x_sq = uniform_filter1d(xi ** 2, win_size, mode='nearest') - mu_x ** 2
        sigma_y_sq = uniform_filter1d(yi ** 2, win_size, mode='nearest') - mu_y ** 2
        sigma_xy = uniform_filter1d(xi * yi, win_size, mode='nearest') - mu_x * mu_y

        # 计算分子和分母
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

        # 防除零处理
        denominator += 1e-10

        # 计算窗口内 SSIM 值并取均值
        ssim_map = numerator / denominator
        ssim_val = np.mean(ssim_map)

        ssim_all[i] = ssim_val

    # 调整输出形状并计算平均值
    ssim_all = ssim_all.reshape(-1, 1)
    ssim_mean = np.mean(ssim_all)

    return ssim_all, ssim_mean

def built_model(model_type, args):
    """
    构建指定类型的神经网络模型 / Build specified type of neural network model

    参数/Args:
        model_type (str): 模型类型（如'ResUNet', 'CBAM_ResUNet'等） / Model type (e.g. 'ResUNet', 'CBAM_ResUNet')
        args (argparse.Namespace): 模型配置参数 / Model configuration parameters

    返回/Returns:
        model (tf.keras.Model): 构建的Keras模型 / Constructed Keras model
        custom_objects (dict): 自定义层和损失函数字典 / Dictionary of custom layers and loss functions
    """
    if model_type == 'ResUNet':
        model = ResUNet.ResUNet(args.num_features, 1, args)
        custom_objects = None

    elif model_type == 'SA_ResUNet':
        model = SA_ResUNet.saResUNet(args.num_features, 1, args)

    elif model_type == 'SE_ResUNet':
        model = SE_ResUNet.seResUNet(args.num_features, 1, args)
        custom_objects = {
            'se_block': SE_ResUNet.se_block,
            'residual_block': SE_ResUNet.residual_block,
            'encoder_block': SE_ResUNet.encoder_block,
            'decoder_block': SE_ResUNet.decoder_block
        }

    elif model_type == 'CBAM_ResUNet':
        model = CBAM_ResUNet.CbamResUNet(args.num_features, 1, args)
        custom_objects = {
            'channel_attention': CBAM_ResUNet.channel_attention,
            'spatial_attention': CBAM_ResUNet.spatial_attention,
            'cbam_block': CBAM_ResUNet.cbam_block,
            'residual_block': CBAM_ResUNet.residual_block,
            'encoder_block': CBAM_ResUNet.encoder_block,
            'decoder_block': CBAM_ResUNet.decoder_block
        }

    elif model_type == 'ECA_ResUNet':
        model = ECA_ResUNet.ecaResUNet(args.num_features, 1, args)
        custom_objects = {
            'eca_block': ECA_ResUNet.eca_block,
            'residual_block': ECA_ResUNet.residual_block,
            'encoder_block': ECA_ResUNet.encoder_block,
            'decoder_block': ECA_ResUNet.decoder_block
        }

    elif model_type == 'BAM_ResUNet':
        model = BAM_ResUNet.BAM_ResUNet(args.num_features, 1, args)
        custom_objects = {
            'bam_block': BAM_ResUNet.bam_block,
            'residual_block': BAM_ResUNet.residual_block,
            'encoder_block': BAM_ResUNet.encoder_block,
            'decoder_block': BAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_reconstruction':
        model = CSAM_ResUNet.CSAMResUNet_reconstruction(args.num_features, 1, args)
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'StopAfterEpochsDecay': CSAM_ResUNet.StopAfterEpochsDecay,
            'K': tf.keras.backend,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }

    elif model_type == 'ResUNet_classification':
        model = ResUNet.ResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'residual_block': ResUNet.residual_block,
            'encoder_block': ResUNet.encoder_block,
            'decoder_block': ResUNet.decoder_block
        }

    elif model_type == 'SE_ResUNet_classification':
        model = SE_ResUNet.seResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'se_block': SE_ResUNet.se_block,
            'residual_block': SE_ResUNet.residual_block,
            'encoder_block': SE_ResUNet.encoder_block,
            'decoder_block': SE_ResUNet.decoder_block
        }

    elif model_type == 'ECA_ResUNet_classification':
        model = ECA_ResUNet.ecaResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'eca_block': ECA_ResUNet.eca_block,
            'residual_block': ECA_ResUNet.residual_block,
            'encoder_block': ECA_ResUNet.encoder_block,
            'decoder_block': ECA_ResUNet.decoder_block
        }

    elif model_type == 'CBAM_ResUNet_classification':
        model = CBAM_ResUNet.CbamResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'channel_attention': CBAM_ResUNet.channel_attention,
            'spatial_attention': CBAM_ResUNet.spatial_attention,
            'cbam_block': CBAM_ResUNet.cbam_block,
            'residual_block': CBAM_ResUNet.residual_block,
            'encoder_block': CBAM_ResUNet.encoder_block,
            'decoder_block': CBAM_ResUNet.decoder_block
        }

    elif model_type == 'BAM_ResUNet_classification':
        model = BAM_ResUNet.BAM_ResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'bam_block': BAM_ResUNet.bam_block,
            'residual_block': BAM_ResUNet.residual_block,
            'encoder_block': BAM_ResUNet.encoder_block,
            'decoder_block': BAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_classification':
        model = CSAM_ResUNet.CSAMResUNet_classification(args.num_features, 1, args)
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'StopAfterEpochsDecay': CSAM_ResUNet.StopAfterEpochsDecay,
            'K': tf.keras.backend,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_classification2':
        model = CSAM_ResUNet.CSAMResUNet_classification2(args.num_features, 1, args)
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'StopAfterEpochsDecay': CSAM_ResUNet.StopAfterEpochsDecay,
            'K': tf.keras.backend,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }

    return model, custom_objects

def load_custom_objects(model_type):
    """
    加载自定义层和损失函数 / Load custom layers and loss functions

    参数/Args:
        model_type (str): 模型类型 / Model type

    返回/Returns:
        custom_objects (dict): 自定义对象字典 / Dictionary of custom objects
    """
    if model_type == 'ResUNet':
        custom_objects = None

    if model_type == 'ResUNet_classification':
        custom_objects = None

    elif model_type == 'SE_ResUNet':
        custom_objects = {
            'se_block': SE_ResUNet.se_block,
            'residual_block': SE_ResUNet.residual_block,
            'encoder_block': SE_ResUNet.encoder_block,
            'decoder_block': SE_ResUNet.decoder_block
        }

    elif model_type == 'CBAM_ResUNet':
        custom_objects = {
            'channel_attention': CBAM_ResUNet.channel_attention,
            'spatial_attention': CBAM_ResUNet.spatial_attention,
            'cbam_block': CBAM_ResUNet.cbam_block,
            'residual_block': CBAM_ResUNet.residual_block,
            'encoder_block': CBAM_ResUNet.encoder_block,
            'decoder_block': CBAM_ResUNet.decoder_block
        }

    elif model_type == 'ECA_ResUNet':
        custom_objects = {
            'eca_block': ECA_ResUNet.eca_block,
            'residual_block': ECA_ResUNet.residual_block,
            'encoder_block': ECA_ResUNet.encoder_block,
            'decoder_block': ECA_ResUNet.decoder_block
        }

    elif model_type == 'BAM_ResUNet':
        custom_objects = {
            'bam_block': BAM_ResUNet.bam_block,
            'residual_block': BAM_ResUNet.residual_block,
            'encoder_block': BAM_ResUNet.encoder_block,
            'decoder_block': BAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_reconstruction':

        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }


    elif model_type == 'SE_ResUNet_classification':
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'se_block': SE_ResUNet.se_block,
            'residual_block': SE_ResUNet.residual_block,
            'encoder_block': SE_ResUNet.encoder_block,
            'decoder_block': SE_ResUNet.decoder_block
        }

    elif model_type == 'CBAM_ResUNet_classification':
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'channel_attention': CBAM_ResUNet.channel_attention,
            'spatial_attention': CBAM_ResUNet.spatial_attention,
            'cbam_block': CBAM_ResUNet.cbam_block,
            'residual_block': CBAM_ResUNet.residual_block,
            'encoder_block': CBAM_ResUNet.encoder_block,
            'decoder_block': CBAM_ResUNet.decoder_block
        }

    elif model_type == 'ECA_ResUNet_classification':
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'eca_block': ECA_ResUNet.eca_block,
            'residual_block': ECA_ResUNet.residual_block,
            'encoder_block': ECA_ResUNet.encoder_block,
            'decoder_block': ECA_ResUNet.decoder_block
        }

    elif model_type == 'BAM_ResUNet_classification':
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'bam_block': BAM_ResUNet.bam_block,
            'residual_block': BAM_ResUNet.residual_block,
            'encoder_block': BAM_ResUNet.encoder_block,
            'decoder_block': BAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_classification':
        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }

    elif model_type == 'CSAM_ResUNet_classification2':

        custom_objects = {
            'DynamicHybridPhysicsLoss': CSAM_ResUNet.DynamicHybridPhysicsLoss,
            'CSAM_block': CSAM_ResUNet.CSAM_block,
            'channel_attention': CSAM_ResUNet.channel_attention,
            'spatial_attention': CSAM_ResUNet.spatial_attention,
            'residual_block': CSAM_ResUNet.residual_block,
            'encoder_block': CSAM_ResUNet.encoder_block,
            'decoder_block': CSAM_ResUNet.decoder_block
        }

    return custom_objects



