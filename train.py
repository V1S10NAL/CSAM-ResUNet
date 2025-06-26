"""
用于训练
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
from tensorflow.keras.callbacks import LambdaCallback
from sklearn.model_selection import train_test_split
import tools
from tools import SaveAndVideoCallback

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.setrecursionlimit(1000)

# 清理GPU内存
# device = cuda.get_current_device()
# device.reset()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定0号GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 程序最多只能占用指定gpu99%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=1024, help='spectral length')
parser.add_argument('--train_data_ratio', type=float, default=0.8, help='training set ratio')   #7:1:2
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--seed', type=int, default=19580920, help='global seed')
parser.add_argument('--net_list', type=str, default=['CSAM_ResUNet_reconstruction'], help='net_list')
##Select the model here [''ResUNet','SE_ResUNet','BAM_ResUNet','CBAM_ResUNet','ECA_ResUNet','CSAM_ResUNet_reconstruction' ]
parser.add_argument('--video', type=bool, default=True, help='generate video')  #


#############################################################


## Load data
start_time = time.perf_counter()
train_path = '.\\dataset\\dataset_train_20000.mat'
extra_path = '.\\dataset\\dataset_extra_10000.mat'
spectra_numpy_array = tools.load_mat_to_np(train_path)
simulated_spectra = spectra_numpy_array['all_simulate_spectra']
original_spectra = spectra_numpy_array['all_original_spectra']

end_time = time.perf_counter()
loading_time = end_time - start_time
print(f"loading time:{loading_time:.6f}")
num_spectra = simulated_spectra.shape[0]

args, unknown = parser.parse_known_args()
parser.add_argument('--num_spectra', type=int, default=num_spectra, help="number of spectra")
args = parser.parse_args()

# 设置随机种子
seed = args.seed
tools.set_seed(args.seed)

X = np.expand_dims(simulated_spectra, axis=-1)
y = np.expand_dims(original_spectra, axis=-1)
# 生成权重矩阵：使用平方函数增强高光强区域的权重
w = 1.0 + np.square(np.square(y))*100  # sample_weight权重 = 1 + 100*y^4，根据 y_true 计算权重根据 y_true 计算权重
# 划分数据集：7:2:1（训练集:验证集:测试集）
X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(X, y, w, test_size=0.3, random_state=seed,
                                                                     shuffle=True)  # 70% 训练集，30% 临时集，分割打乱
X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(X_temp, y_temp, w_temp, test_size=1 / 3,
                                                               random_state=seed,
                                                               shuffle=True)  # 20% 验证集，10% 测试集，打乱

# 定义模型参数
num_outputs = args.num_features  # 输出维度（与目标数据的特征数量一致）
learning_rate = args.learning_rate  # 初始学习率
num_features = args.num_features # 光谱的特征数量/长度
batch_size = args.batch_size # 批大小

for l in range(len(args.net_list)):
    net = args.net_list[l]

    global current_time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = './run/'  + net + '_' + current_time
    result_path = run_path + '/result'
    checkpoint_path = run_path + '/checkpoint'
    movie_path = run_path + '/movie'
    os.makedirs(result_path)
    os.makedirs(checkpoint_path)
    os.makedirs(movie_path)

    print('model: ',net)
    # 创建模型实例
    start_time = time.perf_counter()
    model, custom_objects = tools.built_model(model_type=net, args=args)

    print('model flops:')
    x = tf.constant(np.random.randn(1, 1024, 1))
    print(tools.get_flops(model, [x]))
    print('MFLOPs (Single data) \n')

    print(tools.calculate_flops(model, batch_size= batch_size))
    print('GFLOPs (All data of the batch)\n')

    # X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(X, y, w, test_size=0.3, random_state=seed,
    #                                                                      shuffle=True)  # 70% 训练集，30% 临时集，分割打乱
    # X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(X_temp, y_temp, w_temp, test_size=1 / 3,
    #                                                                random_state=seed,
    #                                                                shuffle=True)  # 20% 验证集，10% 测试集，打乱


    # 定义打印学习率的回调函数
    X_data = X_train[0:1, :, :]
    print_lr_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f" Epoch {epoch + 1}: Learning Rate = {model.optimizer.lr(model.optimizer.iterations).numpy()}"
        )
    )

    video_callback = SaveAndVideoCallback(
        X_data=X_data,
        wavenumber=np.linspace(132.57, 4051.96, 1024),
        output_dir=run_path,
        generate_video=args.video,  # 控制开关
        fps=15,
        sample_idx=0
    )


    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # 使用验证集进行模型评估
        batch_size=batch_size,
        sample_weight=w_train,
        epochs=args.epochs,  # 训练轮数
        verbose=1,  # 打印训练过程
        callbacks=[print_lr_callback, video_callback] # 打印当前学习率，生成视频, update_loss_callback
    )
    end_time = time.perf_counter()
    running_time = end_time - start_time
    print(f"running time:{running_time:.6f}")

    model.save(checkpoint_path + '/' + net + '_model.h5')
    model_save_path = checkpoint_path + '/' + net + '_model.h5'

    # 保存训练过程中的 loss, mae, mape, mse
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    train_mape = history.history['mape']
    val_mape = history.history['val_mape']
    train_mse = history.history['mse']
    val_mse = history.history['val_mse']
    train_cs = history.history['cosine_similarity']
    val_cs = history.history['val_cosine_similarity']
    train_log_cosh = history.history['log_cosh_error']
    val_log_cosh = history.history['val_log_cosh_error']


    # 将数据保存到 DataFrame
    data = {
        'Epoch': range(1, len(train_loss) + 1),
        'Train Loss': train_loss,
        'Validation Loss': val_loss,
        'Train MAE': train_mae,
        'Validation MAE': val_mae,
        'Train MAPE': train_mape,
        'Validation MAPE': val_mape,
        'Train MSE': train_mse,
        'Validation MSE': val_mse,
        'Train CS': train_cs,
        'Validation CS': val_cs,
        'Train log_cosh': train_log_cosh,
        'Validation log_cosh': val_log_cosh

    }
    df = pd.DataFrame(data)

    # 保存到 Excel 文件
    history_file = result_path + "\history.xlsx"
    df.to_excel(history_file, index=False)


    # 绘制训练损失和评估指标
    fig1 = plt.figure(figsize=(12, 5))

    # 绘制损失

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制评估指标
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    # 保存图片
    output_image_path = result_path+"/Loss and MAE.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

    # 使用模型进行预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)


    # 将训练集、验证集和测试集的 X 和 y 转换为 DataFrame
    df_X_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))  # 将 [n, 1024, 1] 转换为 [n, 1024]
    df_y_train = pd.DataFrame(y_train.reshape(y_train.shape[0], -1))
    df_X_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
    df_y_val = pd.DataFrame(y_val.reshape(y_val.shape[0], -1))
    df_X_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
    df_y_test = pd.DataFrame(y_test.reshape(y_test.shape[0], -1))
    df_y_train_pred = pd.DataFrame(y_train_pred.reshape(y_train_pred.shape[0], -1))
    df_y_val_pred = pd.DataFrame(y_val_pred.reshape(y_val_pred.shape[0], -1))
    df_y_test_pred = pd.DataFrame(y_test_pred.reshape(y_test_pred.shape[0], -1))
    # 保存为 CSV 文件
    df_X_train.to_csv(result_path+'/X_train.csv', index=False)
    df_y_train.to_csv(result_path+'/test_data/y_train.csv', index=False)
    df_X_val.to_csv(result_path+'/test_data/X_val.csv', index=False)
    df_y_val.to_csv(result_path+'/test_data/y_val.csv', index=False)
    df_X_test.to_csv(result_path+'/test_data/X_test.csv', index=False)
    df_y_test.to_csv(result_path+'/test_data/y_test.csv', index=False)
    df_y_train_pred.to_csv(result_path+'/test_data/y_train_pred.csv', index=False)
    df_y_val_pred.to_csv(result_path+'/test_data/y_val_pred.csv', index=False)
    df_y_test_pred.to_csv(result_path+'/test_data/y_test_pred.csv', index=False)

    # 绘制第一行的 X_test, y_test, y_test_pred 对比图
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(wavenumber, X_train[10], label='original spectra (X_train)',linewidth=2, alpha=0.7) #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, y_train[10], label='pure spectra (y_train)', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, y_train_pred[10], label='spectra processed by machine learning (y_train_pred)', linewidth=2, alpha=0.9)
    plt.title('Train Comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    output_image_path = result_path+"/train.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(wavenumber, X_val[10], label='original spectra (X_val)',linewidth=2, alpha=0.7) #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, y_val[10], label='pure spectra (y_val)', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, y_val_pred[10], label='spectra processed by machine learning (y_val_pred)', linewidth=2, alpha=0.9)
    plt.title('Val Comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    output_image_path = result_path + "/val.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(wavenumber, X_test[10], label='original spectra (X_test)',linewidth=2, alpha=0.7) #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, y_test[10], label='pure spectra (y_test)', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, y_test_pred[10], label='spectra processed by machine learning (y_test_pred)', linewidth=2, alpha=0.9)
    plt.title('Test Comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    output_image_path = result_path + "/test.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # 计算训练集、验证集和测试集的指标
    print("computing...\n")
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.reshape(y_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    y_val = y_val.reshape(y_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    y_train_pred = y_train_pred.reshape(y_train_pred.shape[0], -1)
    y_val_pred = y_val_pred.reshape(y_val_pred.shape[0], -1)
    y_test_pred = y_test_pred.reshape(y_test_pred.shape[0], -1)
    train_r2, train_mse, train_rmse, train_mae, train_mape = tools.calculate_metrics(y_train, y_train_pred)
    val_r2, val_mse, val_rmse, val_mae, val_mape = tools.calculate_metrics(y_val, y_val_pred)
    test_r2, test_mse, test_rmse, test_mae, test_mape = tools.calculate_metrics(y_test, y_test_pred)



    # 打开文件并写入指标
    metrics_file_name = result_path+"/metrics.txt"
    with open(metrics_file_name, "w", encoding="utf-8") as file:
        file.write(f"Time:{current_time}\n")
        file.write(f"\nModel: {net}\n")
        file.write(f"\nSeed: {seed}\n")
        file.write(f"\nTraining time: {running_time:.4f} seconds\n")
        file.write(f"\nLearning rate: {learning_rate}\n")
        file.write(f"\nBatch size: {batch_size}\n")
        file.write("\nTraining Set Metrics:\n")
        file.write(f"R²: {train_r2:.8f}, MSE: {train_mse:.8f}, RMSE: {train_rmse:.8f}, MAE: {train_mae:.8f}, MAPE: {train_mape:.8f}%\n")
        file.write("\nValidation Set Metrics:\n")
        file.write(f"R²: {val_r2:.8f}, MSE: {val_mse:.8f}, RMSE: {val_rmse:.8f}, MAE: {val_mae:.8f}, MAPE: {val_mape:.8f}%\n")
        file.write("\nTest Set Metrics:\n")
        file.write(f"R²: {test_r2:.8f}, MSE: {test_mse:.8f}, RMSE: {test_rmse:.8f}, MAE: {test_mae:.8f}, MAPE: {test_mape:.8f}%\n")

    # 打印指标
    print("")
    print("------------------------")
    print("Training Set Metrics:")
    print(f"R²: {train_r2:.8f}, MSE: {train_mse:.8f}, RMSE: {train_rmse:.8f}, MAE: {train_mae:.8f}, MAPE: {train_mape:.8f}%")
    print("\nValidation Set Metrics:")
    print(f"R²: {val_r2:.8f}, MSE: {val_mse:.8f}, RMSE: {val_rmse:.8f}, MAE: {val_mae:.8f}, MAPE: {val_mape:.8f}%")
    print("\nTest Set Metrics:")
    print(f"R²: {test_r2:.8f}, MSE: {test_mse:.8f}, RMSE: {test_rmse:.8f}, MAE: {test_mae:.8f}, MAPE: {test_mape:.8f}%")

    ######################### 载入extra数据##################################################
    spectra_numpy_array = tools.load_mat_to_np(extra_path)  ##########改路径

    extra_simulated_spectra = spectra_numpy_array['all_simulate_spectra']  # 模拟光谱
    extra_original_spectra = spectra_numpy_array['all_original_spectra']  # 纯光谱
    extra_num_spectra =  extra_simulated_spectra.shape[0]  # 光谱数

    X_extra = np.expand_dims(extra_simulated_spectra, axis=-1)
    y_extra = np.expand_dims(extra_original_spectra, axis=-1)
    # 绘制第一行的 X_test, y_test, y_test_pred 对比图
    # 模型拟合
    y_extra_pred = model.predict(X_extra)

    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig2 = plt.figure(figsize=(12, 6))

    index_extra = 766;
    plt.plot(wavenumber, X_extra[index_extra], label='original spectra (X_reality)', linewidth=2,
             alpha=0.7)  # linewidth 线宽，alpha透明的
    plt.plot(wavenumber, y_extra[index_extra], label='pure spectra (y_extra)', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, y_extra_pred[index_extra], label='spectra processed by machine learning (y_reality_pred)', linewidth=2,
             alpha=0.9)
    plt.title(' Fit extra Comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    # 非阻塞显示图形
    output_image_path = result_path+"/fit extra.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()


    # 计算性能指标
    X_extra = X_extra.reshape(X_extra.shape[0], -1)
    y_extra = y_extra.reshape(y_extra.shape[0], -1)
    y_extra_pred = y_extra_pred.reshape(y_extra_pred.shape[0], -1)
    extra_r2, extra_mse, extra_rmse, extra_mae, extra_mape = tools.calculate_metrics(y_extra, y_extra_pred)
    # 打开文件并写入指标
    metrics_file_name = result_path + "/Fit extra metrics.txt"
    with open(metrics_file_name, "w", encoding="utf-8") as file:
        file.write("Extra Set Metrics:\n")
        file.write(
            f"R²: {extra_r2:.8f}, MSE: {extra_mse:.8f}, RMSE: {extra_rmse:.8f}, MAE: {extra_mae:.8f}, MAPE: {extra_mape:.8f}%\n")
    # 打印指标
    print("")
    print("------------------------")
    print("Extra Set Metrics:")
    print(f"R²: {extra_r2:.8f}, MSE: {extra_mse:.8f}, RMSE: {extra_rmse:.8f}, MAE: {extra_mae:.8f}, MAPE: {extra_mape:.8f}%")
