"""
用于拟合额外其他光谱
"""
import math
import time
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import colormaps, pyplot as plt
from datetime import datetime
import os
import sys

from scipy.io import savemat
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import shape
import tools


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.setrecursionlimit(1000)
# 清理GPU内存
# device = cuda.get_current_device()
# device.reset()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定0号GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=1024, help='spectral length')
parser.add_argument('--train_data_ratio', type=float, default=0.8, help='training set ratio')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=19580920, help='global seed')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')

# 载入数据
extra_path = '.\\dataset\\dataset_extra_10000.mat'
spectra_numpy_array = tools.load_mat_to_np(extra_path)
# 载入训练好的模型
# 修改路径选择训练好的模型
run_path = 'CSAM_ResUNet_reconstruction_20250610_102330'                          ##########################改模型 CSAM_ResUNet_reconstructiorun_#
#run_path = 'ResUNet_20250610_112536'
#run_path = 'ECA_ResUNet_20250610_170220'

extra_simulated_spectra = spectra_numpy_array['all_simulate_spectra']    # 模拟光谱
extra_original_spectra = spectra_numpy_array['all_original_spectra']         # 纯光谱
extra_bl_spectra = spectra_numpy_array['all_bl_spectra']                           # 带基线光谱
baseline = extra_bl_spectra - extra_original_spectra
num_spectra = extra_simulated_spectra.shape[0]                                       # 光谱数

args, unknown = parser.parse_known_args()
parser.add_argument('--num_spectra', type=int, default=num_spectra, help="number of spectra")
args = parser.parse_args()

# 设置随机种子
seed = args.seed
tools.set_seed(seed)
X_extra= np.expand_dims(extra_simulated_spectra, axis=-1)
y_extra= np.expand_dims(extra_original_spectra, axis=-1)

# X_reality= simulated_spectra
# y_extra= original_spectra
# 载入训练好的模型
# 修改路径选择训练好的模型


parts = run_path.split('_')
if len(parts) >= 2:
    net_name = '_'.join(parts[:-2])  # 去除最后两个分段
else:
    net_name = run_path  # 没有足够分段时保留原字符串
dataset = run_path

# custom_object = tools.load_custom(net_name)
model_path = './run/' + run_path+ '/checkpoint/'+net_name+'_model.h5'  # 修改路径选择训练好的模型

custom_objects = tools.load_custom_objects(net_name)
model = tf.keras.models.load_model(model_path,
                                   custom_objects=custom_objects,
                                   compile=False  # 加载后重新编译)
                                   )
# 固定BN层状态
model.trainable = False   # 设置为推理模式
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model.compile()  # 重新编译以应用设置


global current_time
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
run_path = './run/' + net_name+'_extra' + '_' + current_time
result_path = run_path + '/result'
os.makedirs(result_path)

# 模型拟合
y_extra_pred = model.predict(X_extra)
df_X_extra = pd.DataFrame(X_extra.reshape(X_extra.shape[0],-1))
df_y_extra_pred = pd.DataFrame(y_extra_pred.reshape(y_extra_pred.shape[0], -1))
df_X_extra.to_csv(result_path+'/X_extra.csv', index=False)
df_y_extra_pred.to_csv(result_path+'/y_reality_pred.csv', index=False)

# 绘制第一行的 X_test, y_test, y_test_pred 对比图
wavenumber = np.linspace(132.57, 4051.96, 1024)
fig2 = plt.figure(figsize=(12, 6))

index_extra = 766;
plt.plot(wavenumber, X_extra[index_extra], label='original spectra (X_reality)', linewidth=2, alpha=0.7)  # linewidth 线宽，alpha透明的
plt.plot(wavenumber, y_extra[index_extra], label='pure spectra (y_extra)', linewidth=2, alpha=0.7)
plt.plot(wavenumber, y_extra_pred[index_extra], label='spectra processed by machine learning (y_reality_pred)', linewidth=2,
         alpha=0.9)
plt.title(' extra Comparison')
plt.xlabel('Raman shift')
plt.ylabel('Intensity')
plt.legend()
plt.show()

# 计算性能指标
X_extra = X_extra.reshape(X_extra.shape[0], -1)
y_extra = y_extra.reshape(y_extra.shape[0], -1)
y_extra_pred = y_extra_pred.reshape(y_extra_pred.shape[0], -1)

extra_r2, extra_mse, extra_rmse, extra_mae, extra_mape = tools.calculate_metrics(y_extra, y_extra_pred)
# 打开文件并写入指标
metrics_file_name = result_path + "/metrics.txt"
with open(metrics_file_name, "w", encoding="utf-8") as file:
    file.write("extrating Set Metrics:\n")
    file.write(f"R²: {extra_r2:.8f}, MSE: {extra_mse:.8f}, RMSE: {extra_rmse:.8f}, MAE: {extra_mae:.8f}, MAPE: {extra_mape:.8f}%\n")
# 打印指标
print("")
print("------------------------")
print("extra Set Metrics:")
print(f"R²: {extra_r2:.8f}, MSE: {extra_mse:.8f}, RMSE: {extra_rmse:.8f}, MAE: {extra_mae:.8f}, MAPE: {extra_mape:.8f}%")


# snr_0, snr_0_mean = tools.snr(extra_simulated_spectra,extra_bl_spectra) # 去噪前信噪比
# snr_nn, snr_nn_mean = tools.snr(y_extra_pred+baseline, extra_bl_spectra) # 去噪后信噪比
# psnr_nn, psnr_nn_mean = tools.peak_signal_noise_ratio(y_extra_pred+baseline,extra_bl_spectra) # 去噪后对比纯光谱
# cv_nn, cv_nn_mean = tools.cv(y_extra_pred+baseline) # 去噪后光谱
# cv_0, cv_0_mean = tools.cv(extra_simulated_spectra) # 去噪前光谱
# mse_nn, mse_nn_mean = tools.calculate_mse(y_extra_pred+baseline, extra_bl_spectra) # 对比去噪前后
# ssim_nn, ssim_nn_mean = tools.ssim_1d(y_extra_pred+baseline, extra_bl_spectra, 3) # 对比去噪前后

snr_0, snr_0_mean = tools.snr(extra_original_spectra,extra_simulated_spectra) # 去噪前信噪比
snr_nn, snr_nn_mean = tools.snr(extra_original_spectra, y_extra_pred) # 去噪后信噪比
snr_add_nn = snr_nn - snr_0
snr_add_nn_mean = snr_nn_mean - snr_0_mean

cv_0, cv_0_mean = tools.cv(extra_simulated_spectra) # 去噪前光谱
cv_nn, cv_nn_mean = tools.cv(y_extra_pred) # 去噪后光谱
mrf_0, mrf_0_mean = tools.mrf(extra_simulated_spectra) # 去噪前光谱
mrf_nn, mrf_nn_mean = tools.mrf(y_extra_pred) # 去噪前光谱
psnr_nn, psnr_nn_mean = tools.peak_signal_noise_ratio(y_extra_pred,extra_original_spectra) # 去噪后对比纯光谱
mse_nn, mse_nn_mean = tools.calculate_mse(y_extra_pred, extra_original_spectra) # 对比去噪前后
ssim_nn, ssim_nn_mean = tools.ssim_1d(y_extra_pred, extra_original_spectra, 3) # 对比去噪前后

containers_matrix ={
    'raw' : X_extra,
    'target': y_extra,
    'processed': y_extra_pred,
    'snr_0_nn': snr_0,
    'snr_nn': snr_nn,
    'cv_nn': cv_nn,
    'cv_0_nn': cv_0,
    'mrf_nn': mrf_nn,
    'mrf_0_nn': mrf_0,
    'psnr_nn': psnr_nn,
    'mse_nn': mse_nn,
    'ssim_nn' : ssim_nn,

    'snr_0_nn_mean': snr_0_mean,
    'snr_nn_mean': snr_nn_mean,
    'cv_0_nn_mean': cv_0_mean,
    'cv_nn_mean': cv_nn_mean,
    'mrf_0_nn_mean': mrf_0_mean,
    'mrf_nn_mean': mrf_nn_mean,
    'psnr_nn_mean': psnr_nn_mean,
    'mse_nn_mean': mse_nn_mean,
    'ssim_nn_mean':ssim_nn_mean
}
# 保存文件
out_path =  result_path+'/reconstruction_'+dataset+'_matrix.mat'
save_dict = {
    **containers_matrix,
}

savemat(
    out_path,
    save_dict,
    oned_as='column'      # 确保一维数组保存为列向量
)

print(f"Data successfully saved to: ",out_path)

containers_matrix2 ={
    'snr_0_nn': snr_0,
    'snr_nn': snr_nn,
    'cv_nn': cv_nn,
    'cv_0_nn': cv_0,
    'mrf_nn': mrf_nn,
    'mrf_0_nn': mrf_0,
    'psnr_nn': psnr_nn,
    'mse_nn': mse_nn,
    'ssim_nn' : ssim_nn,

    'snr_0_nn_mean': snr_0_mean,
    'snr_nn_mean': snr_nn_mean,
    'cv_0_nn_mean': cv_0_mean,
    'cv_nn_mean': cv_nn_mean,
    'mrf_0_nn_mean': mrf_0_mean,
    'mrf_nn_mean': mrf_nn_mean,
    'psnr_nn_mean': psnr_nn_mean,
    'mse_nn_mean': mse_nn_mean,
    'ssim_nn_mean':ssim_nn_mean
}
# 打印字典内容
for key, value in containers_matrix2.items():
    print(f"{key}:")
    if isinstance(value, np.ndarray):  # 如果是 NumPy 数组
        print(np.array2string(value, separator=", "))
    else:  # 如果是标量或列表
        print(value)
    print()  # 打印空行分隔



# 保存到 txt 文件
with open(result_path+"\output.txt", "w") as file:
    for key, value in containers_matrix2.items():
        # 将值转换为字符串
        if isinstance(value, np.ndarray):  # 如果是 NumPy 数组
            value_str = np.array2string(value, separator=", ")
        else:  # 如果是标量或列表
            value_str = str(value)
        # 写入文件
        file.write(f"{key}:\n{value_str}\n\n")


# 创建Grad-CAM模型，获取目标层和输出
# 步骤1：选择正确的卷积层（最后一层卷积层）
conv_name = None
for layer in reversed(model.layers):  # 反向遍历，找到最后一层卷积层
    if 'conv1d' in layer.name:
        conv_name = layer.name
        break
print(f"Selected layer: {conv_name}")

# 步骤2：构建梯度模型
target_layer = model.get_layer(conv_name)
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[target_layer.output, model.output]
)

# 步骤3：选择样本并准备数据
sample_idx = 224
sample_input = X_extra[sample_idx].reshape(1, 1024, 1)  # 测试集样本
sample_label = y_extra[sample_idx]

# 步骤4：计算梯度
# 将输入转换为 tf.Variable，以便可以被 tape.watch() 监控
sample_input_var = tf.Variable(sample_input, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(sample_input_var)  # 监控输入
    conv_outputs, predictions = grad_model(sample_input_var)

    # 损失函数：MSE 与目标的差异（假设目标是 sample_label1）
    loss = tf.reduce_sum((predictions - sample_label.reshape(1, 1024, 1)) ** 2)

# 计算梯度
grads = tape.gradient(loss, conv_outputs)  # 对卷积层输出求梯度

# 步骤5：处理梯度和卷积输出
# 获取第一个样本的梯度和卷积输出
conv_outputs = conv_outputs[0]  # (time_steps, filters)
grads_val = grads[0]  # (time_steps, filters)

# 计算每个滤波器的权重（对时间步取均值）
pooled_grads = tf.reduce_mean(grads_val, axis=0)  # 形状 (filters,)

# 步骤6：生成热力图
heatmap = tf.reduce_sum(conv_outputs * pooled_grads[tf.newaxis, :], axis=-1)
heatmap = tf.maximum(heatmap, 0)  # ReLU
heatmap /= tf.reduce_max(heatmap)  # 归一化

# 步骤7：调整热力图分辨率到原始输入长度（如 1024）
# 如果卷积层输出的时间步 < 1024（如 256），则插值
if heatmap.shape[0] != 1024:
    heatmap = tf.image.resize(
        heatmap[tf.newaxis, :, tf.newaxis],  # 添加 batch 和 channel 维度
        [1024, 1],  # 目标尺寸
        method='bilinear'
    )[0, :, 0]  # 移除额外维度

# 步骤8：可视化
plt.figure(figsize=(12, 6))

# 输入光谱
plt.subplot(2, 1, 1)
plt.plot(sample_input[0, :, 0], label='Input Spectrum', color='blue', alpha=0.6)
plt.title('Input Spectrum and Grad-CAM Heatmap')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()

# 热力图叠加
plt.subplot(2, 1, 2)
plt.plot(sample_input[0, :, 0], color='gray', alpha=0.3, label='Input')
plt.plot(heatmap.numpy(), label='Grad-CAM Heatmap', color='red', alpha=0.6)
plt.title('Grad-CAM Heatmap Overlay')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()

gradcam_file_name = result_path + "\gradcam.csv"
df = pd.DataFrame({
    "index": sample_idx+1,
    "Input Spectrum Intensity": sample_input[0, :, 0],
    "Grad-CAM Heatmap Intensity": heatmap.numpy()
})
df.to_csv(gradcam_file_name, index=False)