"""
用于预测真实光谱
"""
import time
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import os
import sys
from scipy.io import savemat
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
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--dataset', type=str, default='test_data', help='dataset')  # 此处选择数据集

# 设置随机种子
seed = 19580920
tools.set_seed(seed)
# 载入数据

path_list_un = [
    '.\\dataset\\dataset_mixture_raw\\dataset1_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset2_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset3_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset4_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset5_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset6_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset7_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset8_norm.mat',
    '.\\dataset\\dataset_mixture_raw\\dataset9_norm.mat',
]

path_list = [
    '.\\dataset\\dataset_150mW_5000ms.mat'
]

path_list = path_list

for path in path_list:

    # run_path = 'CSAM_ResUNet_reconstruction_20250523_182640'                          ##########################改模型#改模型 CSAM_ResUNet_reconstruction
    run_path = 'CSAM_ResUNet_reconstruction_20250610_102330'

    dataset_path = path

    parts_path = dataset_path.split("\\")
    parts_dataset = parts_path[-1].split(".")
    dataset = parts_dataset[0]

    spectra_numpy_array = tools.load_mat_to_np(dataset_path)

    raw_spectra = spectra_numpy_array['raw_spectra']    # 实测光谱

    num_spectra = raw_spectra.shape[0]                                       # 光谱数

    args, unknown = parser.parse_known_args()
    if not hasattr(args, 'num_spectra') or args.num_spectra is None:
        parser.add_argument('--num_spectra', type=int, default=num_spectra, help="number of spectra")
        args = parser.parse_args()

    X_reality= np.expand_dims(raw_spectra, axis=-1)

    # X_reality= simulated_spectra

    # 载入训练好的模型
    # 修改路径选择训练好的模型



    model_date = run_path
    parts = run_path.split('_')
    if len(parts) >= 2:
        net_name = '_'.join(parts[:-2])  # 去除最后两个分段
    else:
        net_name = run_path  # 没有足够分段时保留原字符串
    # dataset = run_path

    # custom_object = tools.load_custom(net_name)
    model_path = './run/' + run_path+ '/checkpoint/'+net_name+'_model.h5'  # 修改路径选择训练好的模型

    custom_objects = tools.load_custom_objects(net_name)
    model = tf.keras.models.load_model(model_path,
                                       # custom_objects=custom_object,
                                       compile=False  # 加载后重新编译)
                                       )
    # 固定BN层状态（关键！）
    model.trainable = False   # 设置为推理模式
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile()  # 重新编译以应用设置


    global current_time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = './run/' + net_name+'_pred' + '_' + current_time
    result_path = run_path + '/result'
    os.makedirs(result_path)

    # 模型拟合
    start_time = time.perf_counter()
    y_reality_pred = model.predict(X_reality)
    end_time = time.perf_counter()
    running_time = end_time - start_time
    per_time = running_time/num_spectra
    print(f"running time:{running_time:.6f}")
    print(f"per spectrum time:{per_time:.6f}")

    df_y_fit_pred = pd.DataFrame(y_reality_pred.reshape(y_reality_pred.shape[0], -1))
    df_y_fit_pred.to_csv(result_path+'/y_reality_pred.csv', index=False)

    # 绘制第一行的 X_test, y_test, y_test_pred 对比图
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig2 = plt.figure(figsize=(12, 6))

    index_fit = 0;
    plt.plot(wavenumber, X_reality[index_fit], label='raw spectra (X_reality)', linewidth=2, alpha=0.7)  # linewidth 线宽，alpha透明的
    plt.plot(wavenumber, y_reality_pred[index_fit], label='spectra processed by machine learning (y_reality_pred)', linewidth=2,
             alpha=0.9)
    plt.title(' Machine learning processing')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show(block=False)
    plt.pause(1)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close(fig2)

    # 计算性能指标
    X_reality = X_reality.reshape(X_reality.shape[0], -1)
    y_reality_pred = y_reality_pred.reshape(y_reality_pred.shape[0], -1)

    # 将 NumPy 数组转换为 pandas DataFrame
    df_X_reality = pd.DataFrame(X_reality)
    df_y_reality_pred = pd.DataFrame(y_reality_pred)

    # 保存为 Excel 文件
    df_X_reality.to_csv(result_path+'/'+dataset +'_X_reality.csv', index=False)
    df_y_reality_pred.to_csv(result_path+'/'+dataset+'_y_reality_pred.csv', index=False)



    cv_0, cv_0_mean = tools.cv(X_reality) # 去噪前光谱
    cv_nn, cv_nn_mean = tools.cv(y_reality_pred) # 去噪后光谱
    mrf_0, mrf_0_mean = tools.mrf(X_reality) # 去噪前光谱
    mrf_nn, mrf_nn_mean = tools.mrf(y_reality_pred) # 去噪前光谱
    snr_nn, snr_tc_mean =tools.snr(X_reality,y_reality_pred)
    psnr_nn, psnr_nn_mean = tools.peak_signal_noise_ratio(y_reality_pred, X_reality) # 去噪后对比纯光谱
    mse_nn, mse_nn_mean = tools.calculate_mse(y_reality_pred, X_reality) # 对比去噪前后
    ssim_nn, ssim_nn_mean = tools.ssim_1d(y_reality_pred, X_reality, 3) # 对比去噪前后

    containers_matrix ={
        'raw': X_reality,
        'processed': y_reality_pred,

        'running_time': running_time,
        'per_time': per_time,
        'cv_nn': cv_nn,
        'cv_0_nn': cv_0,
        'mrf_nn':mrf_nn,
        'mrf_0_nn':mrf_0,
        'psnr_nn': psnr_nn,
        'mse_nn': mse_nn,
        'ssim_nn' : ssim_nn,

        'cv_0_nn_mean': cv_0_mean,
        'cv_nn_mean': cv_nn_mean,
        'mrf_0_nn_mean': mrf_0_mean,
        'mrf_nn_mean': mrf_nn_mean,
        'psnr_nn_mean': psnr_nn_mean,
        'mse_nn_mean': mse_nn_mean,
        'ssim_nn_mean':ssim_nn_mean
    }

    containers_matrix2 ={
        'running_time': running_time,
        'per_time': per_time,

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

    # 保存到mat
    out_path =  result_path+'/predict_'+model_date+'_'+dataset+'_matrix.mat'

    save_dict = {
        **containers_matrix,
    }
    savemat(
        out_path,
        save_dict,
        oned_as='column'      # 确保一维数组保存为列向量
    )
    print(f"Data successfully saved to: ",out_path)



