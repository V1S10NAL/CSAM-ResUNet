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
from sklearn.metrics import roc_curve, auc
import tools
from scipy.io import savemat
from sklearn.metrics import classification_report
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.setrecursionlimit(1000)
# 清理GPU内存
# device = cuda.get_current_device()+
# device.reset()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定0号GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--num_features', type=int, default=1024, help='spectral length')
parser.add_argument('--num_outputs', type=int, default=1, help='num_outputs')
parser.add_argument('--train_data_ratio', type=float, default=0.8, help='training set ratio')  #7:1:2
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--net_list', type=str, default=['CSAM_ResUNet_classification'],
                    help='net_list')  # 此处选择模型['ResUNet','SE_ResUNet','ECA_ResUNet','BAM_ResUNet','CBAM_ResUNet','CSAM_ResUNet_classification']
parser.add_argument('--num_substances', type=int, default=5, help='num_substances')
parser.add_argument('--MP_type', type=str, default=['PC', 'PE', 'PP', 'PS', 'PVC'], help='microplastics types')
args = parser.parse_args(args=[])

# 设置随机种子
seed = 19580920

tools.set_seed(seed)

########################################################################################################
## 载入数据
start_time = time.perf_counter()

path_data_train = '.\\dataset\\dataset_train_mixture_36000.mat' ###
path_data_pred= '.\\dataset\\dataset_mixture_denoised\\dataset1_denoised.mat'###
run_path: str = 'CSAM_ResUNet_classification_20250617_085150'  ###############模型路径 CSAM_ResUNet_classification

parts = run_path.split('_')
if len(parts) >= 2:
    net_name = '_'.join(parts[:-2])  # 去除最后两个分段
else:
    net_name = run_path  # 没有足够分段时保留原字符串
dataset = run_path

spectra_mat = tools.load_mat_to_np(path_data_train)  ###dataset_train_mix_1000.mat     dataset_train_mixed_mp.mat

data_vars = [
    'PCPE', 'PCPP', 'PCPS', 'PCPVC',
    'PEPP', 'PEPS', 'PEPVC',
    'PPPS', 'PPPVC',
    'PSPVC'
]

categories = args.MP_type
num_classes = len(categories)
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

# 使用字典存储各材料对应的数据和标签列表
data_dict = {
    'X': {m: [] for m in data_vars},
    'X1': {m: [] for m in data_vars},
    'X2': {m: [] for m in data_vars},
    'y': {m: [] for m in data_vars},

    'X_train': {m: [] for m in data_vars},
    'X_val': {m: [] for m in data_vars},
    'X_test': {m: [] for m in data_vars},
    'X_temp': {m: [] for m in data_vars},
    'X_test_pred': {m: [] for m in data_vars},
    'X_extra': {m: [] for m in data_vars},
    'X_fit_pred': {m: [] for m in data_vars},

    'X1_train': {m: [] for m in data_vars},
    'X1_val': {m: [] for m in data_vars},
    'X1_test': {m: [] for m in data_vars},
    'X1_temp': {m: [] for m in data_vars},
    'X1_test_pred': {m: [] for m in data_vars},
    'X1_fit_pred': {m: [] for m in data_vars},

    'X2_train': {m: [] for m in data_vars},
    'X2_val': {m: [] for m in data_vars},
    'X2_test': {m: [] for m in data_vars},
    'X2_temp': {m: [] for m in data_vars},
    'X2_test_pred': {m: [] for m in data_vars},
    'X2_fit_pred': {m: [] for m in data_vars},

    'y_train': {m: [] for m in data_vars},
    'y_val': {m: [] for m in data_vars},
    'y_test': {m: [] for m in data_vars},
    'y_temp': {m: [] for m in data_vars},
    'y_test_pred': {m: [] for m in data_vars},
    'y_extra': {m: [] for m in data_vars},
    'y_extra_pred': {m: [] for m in data_vars},

    'fit': {m: [] for m in data_vars},
    "metrics": {m: [] for m in data_vars}
}

num_spectra = 0

# 读取数据，生成二维向量标签
X = []
X1 = []
X2 = []
y = []
X_train = []
X1_train = []
X2_train = []
y_train = []
X_val = []
X1_val = []
X2_val = []
y_val = []
X_test = []
X1_test = []
X2_test = []
y_test = []

# 读取数据，生成二维向量标签
for var_name in data_vars:
    data = spectra_mat[var_name]
    print(var_name)
    # 解析标签（前两位和后两位物质）#######################################
    substances = [var_name[:2], var_name[2:]]  # 假设变量名格式如 PCPE
    label = np.zeros(num_classes)
    for sub in substances:
        if sub in category_to_idx:
            label[category_to_idx[sub]] = 1

    data_dict['X'][var_name] = data
    data_dict['y'][var_name].append(np.tile(label, (data.shape[0], 1)))  # 标签重复对应样本数
    # 提取端元#################################################
    mp1 = var_name + '_' + var_name[:2]
    data_mp1 = spectra_mat[mp1]
    data_dict['X1'][var_name].append(data_mp1)
    mp2 = var_name + '_' + var_name[2:]
    data_mp2 = spectra_mat[mp2]
    data_dict['X2'][var_name].append(data_mp2)
    print(label)

    num_spectra = num_spectra + data.shape[0]

    (data_dict['X_train'][var_name],
     data_dict['X_temp'][var_name],
     data_dict['X1_train'][var_name],
     data_dict['X1_temp'][var_name],
     data_dict['X2_train'][var_name],
     data_dict['X2_temp'][var_name],
     data_dict['y_train'][var_name],
     data_dict['y_temp'][var_name]) = train_test_split(
        np.squeeze(data_dict['X'][var_name]),
        np.squeeze(data_dict['X1'][var_name]),
        np.squeeze(data_dict['X2'][var_name]),
        np.squeeze(data_dict['y'][var_name]), test_size=0.3, random_state=seed)

    (data_dict['X_val'][var_name],
     data_dict['X_test'][var_name],
     data_dict['X1_val'][var_name],
     data_dict['X1_test'][var_name],
     data_dict['X2_val'][var_name],
     data_dict['X2_test'][var_name],
     data_dict['y_val'][var_name],
     data_dict['y_test'][var_name]) = train_test_split(
        np.squeeze(data_dict['X_temp'][var_name]),
        np.squeeze(data_dict['X1_temp'][var_name]),
        np.squeeze(data_dict['X2_temp'][var_name]),
        np.squeeze(data_dict['y_temp'][var_name]), test_size=1 / 3,
        shuffle=False,  # 按顺序划分
        #random_state=seed
    )

    X.append(np.squeeze(data_dict['X'][var_name]))
    X1.append(np.squeeze(data_dict['X1'][var_name]))
    X2.append(np.squeeze(data_dict['X2'][var_name]))
    y.append(np.squeeze(data_dict['y'][var_name]))

    X_train.append(np.squeeze(data_dict['X_train'][var_name]))
    X1_train.append(np.squeeze(data_dict['X1_train'][var_name]))
    X2_train.append(np.squeeze(data_dict['X2_train'][var_name]))
    y_train.append(np.squeeze(data_dict['y_train'][var_name]))

    X_val.append(np.squeeze(data_dict['X_val'][var_name]))
    X1_val.append(np.squeeze(data_dict['X1_val'][var_name]))
    X2_val.append(np.squeeze(data_dict['X2_val'][var_name]))
    y_val.append(np.squeeze(data_dict['y_val'][var_name]))

    X_test.append(np.squeeze(data_dict['X_test'][var_name]))
    X1_test.append(np.squeeze(data_dict['X1_test'][var_name]))
    X2_test.append(np.squeeze(data_dict['X2_test'][var_name]))
    y_test.append(np.squeeze(data_dict['y_test'][var_name]))

X = np.vstack(X)
X1 = np.vstack(X1)
X2 = np.vstack(X2)
y = np.vstack(y)

X_train = np.vstack(X_train)
X1_train = np.vstack(X1_train)
X2_train = np.vstack(X2_train)
y_train = np.vstack(y_train)

X_val = np.vstack(X_val)
X1_val = np.vstack(X1_val)
X2_val = np.vstack(X2_val)
y_val = np.vstack(y_val)

X_test = np.vstack(X_test)
X1_test = np.vstack(X1_test)
X2_test = np.vstack(X2_test)
y_test = np.vstack(y_test)
# 合并所有数据


end_time = time.perf_counter()
loading_time = end_time - start_time
print(f"loading time:{loading_time:.6f}")

#num_spectra = spectra_mat[''].shape[0]                                      #光谱数
args, unknown = parser.parse_known_args()
parser.add_argument('--num_spectra', type=int, default=num_spectra, help="number of spectra")
args = parser.parse_args()



X = np.expand_dims(X, axis=-1)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X1 = np.expand_dims(X1, axis=-1)
X1_train = np.expand_dims(X1_train, axis=-1)
X1_val = np.expand_dims(X1_val, axis=-1)
X1_test = np.expand_dims(X1_test, axis=-1)
X2 = np.expand_dims(X2, axis=-1)
X2_train = np.expand_dims(X2_train, axis=-1)
X2_val = np.expand_dims(X2_val, axis=-1)
X2_test = np.expand_dims(X2_test, axis=-1)

# 定义模型参数
num_outputs = args.num_outputs  # 输出维度（与目标数据的特征数量一致）
learning_rate = args.learning_rate  # 初始学习率
num_features = args.num_features  # 光谱的特征数量/长度
batch_size = args.batch_size  # 批大小

for l in range(len(args.net_list)):
    net = args.net_list[l]

    #global current_time
    # current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # run_path = './run/' + net + '_' + current_time
    # result_path = run_path + '/result'
    # checkpoint_path = run_path + '/checkpoint'
    # movie_path = run_path + '/movie'
    # os.makedirs(result_path)
    # os.makedirs(checkpoint_path)
    # os.makedirs(movie_path)
    #
    # print('model: ', net)
    # 创建模型实例
    start_time = time.perf_counter()
    #model = tools.built_model(model_type=net, args=args)##########
    #model, custom_objects = tools.built_model(model_type=net, args=args)

    # print('model flops:')
    # x = tf.constant(np.random.randn(1, 1024, 1))
    # print(tools.get_flops(model, [x]))
    # print('MFLOPs\n')
    #
    # print(tools.calculate_flops(model, batch_size=batch_size))
    # print('GFLOPs\n')
    #
    # # 定义打印学习率的回调函数
    # print_lr_callback = LambdaCallback(
    #     on_epoch_end=lambda epoch, logs: print(
    #         f" Epoch {epoch + 1}: Learning Rate = {model.optimizer.lr(model.optimizer.iterations).numpy()}"
    #     )
    # )

    # 训练模型
    # history = model.fit(
    #     x=X_train,
    #     y={
    #         'spectrum_A': X1_train,
    #         'spectrum_B': X2_train,
    #         'classification': y_train
    #     },
    #     validation_data=(
    #         X_val,
    #         {
    #             'spectrum_A': X1_val,
    #             'spectrum_B': X2_val,
    #             'classification': y_val
    #         }
    #     ),
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,  # 训练轮数
    #     verbose=1,  # 打印训练过程
    #     callbacks=[print_lr_callback]
    # )
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

    end_time = time.perf_counter()
    running_time = end_time - start_time
    print(f"running time:{running_time:.6f}")

    # model.save(checkpoint_path + '/' + net + '_model.h5')
    # model_save_path = checkpoint_path + '/' + net + '_model.h5'

    # 保存训练过程中的 loss, mae, mape, mse
    #
    # train_loss = history.history['classification_loss']
    # val_loss = history.history['val_classification_loss']
    #
    # train_spectrum_a_loss = history.history['spectrum_A_loss']
    # train_spectrum_b_loss = history.history['spectrum_B_loss']
    # val_spectrum_a_loss = history.history['val_spectrum_A_loss']
    # val_spectrum_b_loss = history.history['val_spectrum_B_loss']
    #
    # train_spectrum_a_mse = history.history['spectrum_A_mse']
    # train_spectrum_b_mse = history.history['spectrum_B_mse']
    # val_spectrum_a_mse = history.history['val_spectrum_A_mse']
    # val_spectrum_b_mse = history.history['val_spectrum_B_mse']
    #
    # train_spectrum_a_mae = history.history['spectrum_A_mae']
    # train_spectrum_b_mae = history.history['spectrum_B_mae']
    # val_spectrum_a_mae = history.history['val_spectrum_A_mae']
    # val_spectrum_b_mae = history.history['val_spectrum_B_mae']
    #
    # accuracy = history.history['classification_binary_accuracy']
    # precision = history.history['classification_precision']
    # recall = history.history['classification_recall']
    # val_accuracy = history.history['val_classification_binary_accuracy']
    # val_precision = history.history['val_classification_precision']
    # val_recall = history.history['val_classification_recall']

    # 将数据保存到 DataFrame
    # data = {
    #     'Epoch': range(1, len(train_loss) + 1),
    #
    #     'Train Loss': train_loss,
    #     'Validation Loss': val_loss,
    #
    #     'Train spectrum A Loss': train_spectrum_a_loss,
    #     'Validation  spectrum A Loss': val_spectrum_a_loss,
    #     'Train spectrum B Loss': train_spectrum_b_loss,
    #     'Validation  spectrum B Loss': val_spectrum_b_loss,
    #
    #     'Train spectrum A MSE': train_spectrum_a_mse,
    #     'Validation  spectrum A MSE': val_spectrum_a_mse,
    #     'Train spectrum B MSE': train_spectrum_b_mse,
    #     'Validation  spectrum B MSE': val_spectrum_b_mse,
    #
    #     'Train spectrum A MAE': train_spectrum_a_mae,
    #     'Validation  spectrum A MAE': val_spectrum_b_mae,
    #     'Train spectrum B MAE': train_spectrum_a_mae,
    #     'Validation  spectrum B MAE': val_spectrum_b_mae,
    #
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'val_accuracy': val_accuracy,
    #     'val_precision': val_precision,
    #     'val_recall': val_recall,
    # }
    # df = pd.DataFrame(data)
    #
    # # 保存到 Excel 文件
    # history_file = result_path + "\history.xlsx"
    # df.to_excel(history_file, index=False)

    # ==================== 光谱解耦结果评估 ====================

    # 绘制训练损失和评估指标
    # fig1 = plt.figure(figsize=(12, 4))
    #
    # # 绘制损失
    #
    # plt.subplot(1, 3, 1)
    # plt.plot(history.history['classification_loss'], label='Training Loss')
    # plt.plot(history.history['val_classification_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # 绘制评估指标
    # plt.subplot(1, 3, 2)
    # plt.plot(history.history['spectrum_A_loss'], label='Training Loss')
    # plt.plot(history.history['val_spectrum_A_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Spectrum A Loss')
    # plt.legend()
    #
    # plt.subplot(1, 3, 3)
    # plt.plot(history.history['spectrum_B_loss'], label='Training Loss')
    # plt.plot(history.history['val_spectrum_B_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Spectrum B Loss')
    # plt.legend()
    #
    # # 保存图片
    # output_image_path = result_path + "/Loss.png"  # 定义保存路径和文件名
    # plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # # 非阻塞显示图形
    # plt.show(block=False)
    #
    # # 暂停一段时间（例如 2 秒）
    # plt.pause(2)  # 2 秒后继续执行
    #
    # # 关闭图形窗口
    # plt.close()

    # ========================== 使用模型进行预测 ==========================

    x1_pred, x2_pred, y_pred = model.predict(X)
    x_val_A_pred, x_val_B_pred, y_val_classify_pred = model.predict(X_val)
    x_train_A_pred, x_train_B_pred, y_train_classify_pred = model.predict(X_train)
    x_test_A_pred, x_test_B_pred, y_test_classify_pred = model.predict(X_test)

    # ========================== 每类混合物单独预测 ========================
    for var_name in data_vars:
        # 获取当前物质组合的测试数据
        X_test_var = data_dict['X_test'][var_name]

        # 调整数据形状以匹配模型输入 (添加通道维度)
        X_test_var = np.expand_dims(X_test_var, axis=-1)

        # 执行预测
        X1_test_pred, X2_test_pred, y_test_pred = model.predict(X_test_var)

        # 将预测结果存储到 data_dict 中
        data_dict['X1_test_pred'][var_name] = X1_test_pred
        data_dict['X2_test_pred'][var_name] = X2_test_pred
        data_dict['y_test_pred'][var_name] = y_test_pred

        # 打印进度
        print(f"Processed predictions for {var_name}")

    # 遍历所有物质组合
    var_to_labels = {
        "PCPE": ["PC", "PE"],
        "PCPP": ["PC", "PP"],
        "PCPS": ["PC", "PS"],
        "PCPVC": ["PC", "PVC"],
        "PEPP": ["PE", "PP"],
        "PEPS": ["PE", "PS"],
        "PEPVC": ["PE", "PVC"],
        "PPPS": ["PP", "PS"],
        "PPPVC": ["PP", "PVC"],
        "PSPVC": ["PS", "PVC"],
    }
    for var_name in data_dict['y_test_pred']:
        # 获取真实标签和预测概率
        y_true = data_dict['y_test'][var_name]  # 形状 (n_samples, n_classes)
        y_pred_proba = data_dict['y_test_pred'][var_name]  # 形状 (n_samples, n_classes)

        # 检查数据有效性
        if y_true.shape != y_pred_proba.shape:
            print(f"Shape mismatch for {var_name}, skipping...")
            continue

        # 计算指标（使用Top-2策略）
        metrics = tools.evaluate_multilabel_accuracy(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            class_names=args.MP_type,  # 替换为你的类别名称列表
            var=var_name,
            var_to_labels=var_to_labels,
            type_mp=args.MP_type,
            top_k=2
        )

        # 将结果存入数据字典
        data_dict["metrics"][var_name] = metrics

        # 打印结果
        print(f"\n=== Metrics for {var_name} ===")
        print(f"Subset Accuracy （双标签同时正确）: {metrics['subset_accuracy']:.4f}")
        print(f"Average Sample Accuracy （单标签正确）: {metrics['average_sample_accuracy']:.4f}")
        for label, acc in metrics["label_accuracy"].items():
            print(f"Accuracy  {label}: {acc:.4f}")

    # =============================== 数据保存 ==============================
    # 将训练集、验证集和测试集的 X 和 y 转换为 DataFrame
    df_X_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
    df_X_val = pd.DataFrame(X_val.reshape(X_val.shape[0], -1))
    df_X_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))

    df_x_train_A_pred = pd.DataFrame(x_train_A_pred.reshape(x_train_A_pred.shape[0], -1))
    df_x_val_A_pred = pd.DataFrame(x_val_A_pred.reshape(x_val_A_pred.shape[0], -1))
    df_x_test_A_pred = pd.DataFrame(x_test_A_pred.reshape(x_test_A_pred.shape[0], -1))

    df_x_train_B_pred = pd.DataFrame(x_train_B_pred.reshape(x_train_B_pred.shape[0], -1))
    df_x_val_B_pred = pd.DataFrame(x_val_B_pred.reshape(x_val_B_pred.shape[0], -1))
    df_x_test_B_pred = pd.DataFrame(x_test_B_pred.reshape(x_test_B_pred.shape[0], -1))

    # 保存为 CSV 文件
    # df_X_train.to_csv(result_path + '/X_train.csv', index=False)
    # df_X_train.to_csv(result_path + '/X_val.csv', index=False)
    # df_X_train.to_csv(result_path + '/X_test.csv', index=False)
    #
    # df_x_train_A_pred.to_csv(result_path + '/X_A_pred_train.csv', index=False)
    # df_x_val_A_pred.to_csv(result_path + '/X_A_pred_val.csv', index=False)
    # df_x_test_A_pred.to_csv(result_path + '/X_A_pred_test.csv', index=False)
    #
    # df_x_train_B_pred.to_csv(result_path + '/X_B_pred_train.csv', index=False)
    # df_x_val_B_pred.to_csv(result_path + '/X_B_pred_val.csv', index=False)
    # df_x_test_B_pred.to_csv(result_path + '/X_B_pred_test.csv', index=False)

    # ==================== train ====================
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig2 = plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(wavenumber, X_train[10], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, X1_train[10], label='Spectrum A', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_train[10], label='Spectrum B', linewidth=2, alpha=0.7)

    plt.title('Train target comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(wavenumber, X_train[10], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, x_train_A_pred[10], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, x_train_B_pred[10], label='Spectrum B pred', linewidth=2, alpha=0.7)

    plt.title('Train prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(wavenumber, x_train_A_pred[1], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X1_train[1], label='Spectrum A ', linewidth=2, alpha=0.7)

    plt.title('Train prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(wavenumber, x_train_B_pred[1], label='Spectrum B pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_train[1], label='Spectrum B ', linewidth=2, alpha=0.7)

    plt.title('Train prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    output_image_path = result_path + "/train.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

    # ==================== val ====================
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig3 = plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(wavenumber, X_val[1], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, X1_val[1], label='Spectrum A', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_val[1], label='Spectrum B', linewidth=2, alpha=0.7)

    plt.title('Val target comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(wavenumber, X_val[1], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, x_val_A_pred[1], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, x_val_B_pred[1], label='Spectrum B pred', linewidth=2, alpha=0.7)

    plt.title('Val prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(wavenumber, x_val_A_pred[1], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X1_val[1], label='Spectrum A ', linewidth=2, alpha=0.7)

    plt.title('Val prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(wavenumber, x_val_B_pred[1], label='Spectrum B pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_val[1], label='Spectrum B ', linewidth=2, alpha=0.7)

    plt.title('Val prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    output_image_path = result_path + "/val.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

    # ==================== test ====================
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig4 = plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(wavenumber, X_test[1], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, X1_test[1], label='Spectrum A', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_test[1], label='Spectrum B', linewidth=2, alpha=0.7)

    plt.title('Test target comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(wavenumber, X_test[1], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, x_test_A_pred[1], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, x_test_B_pred[1], label='Spectrum B pred', linewidth=2, alpha=0.7)

    plt.title('Test prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(wavenumber, x_test_A_pred[1], label='Spectrum A pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X1_test[1], label='Spectrum A ', linewidth=2, alpha=0.7)

    plt.title('Test prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(wavenumber, x_test_B_pred[1], label='Spectrum B pred', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, X2_test[1], label='Spectrum B ', linewidth=2, alpha=0.7)

    plt.title('Test prediction comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    output_image_path = result_path + "/test.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

    # 计算训练集、验证集和测试集的指标

    print("computing...\n")
    X1_train = X1_train.reshape(X1_train.shape[0], -1)
    X2_train = X2_train.reshape(X2_train.shape[0], -1)
    X1_val = X1_val.reshape(X1_val.shape[0], -1)
    X2_val = X2_val.reshape(X2_val.shape[0], -1)
    X1_test = X1_test.reshape(X1_test.shape[0], -1)
    X2_test = X2_test.reshape(X2_test.shape[0], -1)

    x_train_A_pred = x_train_A_pred.reshape(x_train_A_pred.shape[0], -1)
    x_val_A_pred = x_val_A_pred.reshape(x_val_A_pred.shape[0], -1)
    x_test_A_pred = x_test_A_pred.reshape(x_test_A_pred.shape[0], -1)
    x_train_B_pred = x_train_B_pred.reshape(x_train_B_pred.shape[0], -1)
    x_val_B_pred = x_val_B_pred.reshape(x_val_B_pred.shape[0], -1)
    x_test_B_pred = x_test_B_pred.reshape(x_test_B_pred.shape[0], -1)

    spectrum_A_train_r2, spectrum_A_train_mse, spectrum_A_train_rmse, spectrum_A_train_mae, spectrum_A_train_mape = tools.calculate_metrics(
        X1_train, x_train_A_pred)
    spectrum_A_val_r2, spectrum_A_val_mse, spectrum_A_val_rmse, spectrum_A_val_mae, spectrum_A_val_mape = tools.calculate_metrics(
        X1_val, x_val_A_pred)
    spectrum_A_test_r2, spectrum_A_test_mse, spectrum_A_test_rmse, spectrum_A_test_mae, spectrum_A_test_mape = tools.calculate_metrics(
        X1_test, x_test_A_pred)
    spectrum_B_train_r2, spectrum_B_train_mse, spectrum_B_train_rmse, spectrum_B_train_mae, spectrum_B_train_mape = tools.calculate_metrics(
        X2_train, x_train_B_pred)
    spectrum_B_val_r2, spectrum_B_val_mse, spectrum_B_val_rmse, spectrum_B_val_mae, spectrum_B_val_mape = tools.calculate_metrics(
        X2_val, x_val_B_pred)
    spectrum_B_test_r2, spectrum_B_test_mse, spectrum_B_test_rmse, spectrum_B_test_mae, spectrum_B_test_mape = tools.calculate_metrics(
        X2_test, x_test_B_pred)

    # 打开文件并写入指标
    metrics_file_name = result_path + "/metrics.txt"
    with open(metrics_file_name, "w", encoding="utf-8") as file:
        file.write(f"Time:{current_time}\n")
        file.write(f"\nModel: {net}\n")
        file.write(f"\nTraining time: {running_time:.4f} seconds\n")
        file.write(f"\nLearning rate: {learning_rate}\n")
        file.write(f"\nBatch size: {batch_size}\n")

        # 写入Spectrum A指标
        file.write("\n\n=== Spectrum A Metrics ===")
        file.write("\n\nTraining Set Metrics:\n")
        file.write(
            f"R²: {spectrum_A_train_r2:.8f}, MSE: {spectrum_A_train_mse:.8f}, RMSE: {spectrum_A_train_rmse:.8f}, MAE: {spectrum_A_train_mae:.8f}, MAPE: {spectrum_A_train_mape:.8f}%\n")
        file.write("\nValidation Set Metrics:\n")
        file.write(
            f"R²: {spectrum_A_val_r2:.8f}, MSE: {spectrum_A_val_mse:.8f}, RMSE: {spectrum_A_val_rmse:.8f}, MAE: {spectrum_A_val_mae:.8f}, MAPE: {spectrum_A_val_mape:.8f}%\n")
        file.write("\nTest Set Metrics:\n")
        file.write(
            f"R²: {spectrum_A_test_r2:.8f}, MSE: {spectrum_A_test_mse:.8f}, RMSE: {spectrum_A_test_rmse:.8f}, MAE: {spectrum_A_test_mae:.8f}, MAPE: {spectrum_A_test_mape:.8f}%\n")

        # 写入Spectrum B指标
        file.write("\n\n=== Spectrum B Metrics ===")
        file.write("\n\nTraining Set Metrics:\n")
        file.write(
            f"R²: {spectrum_B_train_r2:.8f}, MSE: {spectrum_B_train_mse:.8f}, RMSE: {spectrum_B_train_rmse:.8f}, MAE: {spectrum_B_train_mae:.8f}, MAPE: {spectrum_B_train_mape:.8f}%\n")
        file.write("\nValidation Set Metrics:\n")
        file.write(
            f"R²: {spectrum_B_val_r2:.8f}, MSE: {spectrum_B_val_mse:.8f}, RMSE: {spectrum_B_val_rmse:.8f}, MAE: {spectrum_B_val_mae:.8f}, MAPE: {spectrum_B_val_mape:.8f}%\n")
        file.write("\nTest Set Metrics:\n")
        file.write(
            f"R²: {spectrum_B_test_r2:.8f}, MSE: {spectrum_B_test_mse:.8f}, RMSE: {spectrum_B_test_rmse:.8f}, MAE: {spectrum_B_test_mae:.8f}, MAPE: {spectrum_B_test_mape:.8f}%\n")

    # 打印指标
    print("")
    print("------------------------")
    print("=== Spectrum A Metrics ===")
    print("\nTraining Set Metrics:")
    print(
        f"R²: {spectrum_A_train_r2:.8f}, MSE: {spectrum_A_train_mse:.8f}, RMSE: {spectrum_A_train_rmse:.8f}, MAE: {spectrum_A_train_mae:.8f}, MAPE: {spectrum_A_train_mape:.8f}%")
    print("\nValidation Set Metrics:")
    print(
        f"R²: {spectrum_A_val_r2:.8f}, MSE: {spectrum_A_val_mse:.8f}, RMSE: {spectrum_A_val_rmse:.8f}, MAE: {spectrum_A_val_mae:.8f}, MAPE: {spectrum_A_val_mape:.8f}%")
    print("\nTest Set Metrics:")
    print(
        f"R²: {spectrum_A_test_r2:.8f}, MSE: {spectrum_A_test_mse:.8f}, RMSE: {spectrum_A_test_rmse:.8f}, MAE: {spectrum_A_test_mae:.8f}, MAPE: {spectrum_A_test_mape:.8f}%")

    print("\n------------------------")
    print("=== Spectrum B Metrics ===")
    print("\nTraining Set Metrics:")
    print(
        f"R²: {spectrum_B_train_r2:.8f}, MSE: {spectrum_B_train_mse:.8f}, RMSE: {spectrum_B_train_rmse:.8f}, MAE: {spectrum_B_train_mae:.8f}, MAPE: {spectrum_B_train_mape:.8f}%")
    print("\nValidation Set Metrics:")
    print(
        f"R²: {spectrum_B_val_r2:.8f}, MSE: {spectrum_B_val_mse:.8f}, RMSE: {spectrum_B_val_rmse:.8f}, MAE: {spectrum_B_val_mae:.8f}, MAPE: {spectrum_B_val_mape:.8f}%")
    print("\nTest Set Metrics:")
    print(
        f"R²: {spectrum_B_test_r2:.8f}, MSE: {spectrum_B_test_mse:.8f}, RMSE: {spectrum_B_test_rmse:.8f}, MAE: {spectrum_B_test_mae:.8f}, MAPE: {spectrum_B_test_mape:.8f}%\n")

    ## ==================== 光谱分类结果评估 ====================

    # fig5 = plt.figure(figsize=(6, 6))
    # plt.plot(history.history['classification_binary_accuracy'], label='Accuracy')
    # plt.plot(history.history['classification_precision'], label='Precision')
    # plt.plot(history.history['classification_recall'], label='Recall')
    # plt.plot(history.history['val_classification_binary_accuracy'], '--', label='Val Accuracy')
    # plt.plot(history.history['val_classification_precision'], '--', label='Val Precision')
    # plt.plot(history.history['val_classification_recall'], '--', label='Val Recall')
    # plt.title('Classification Metrics')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.tight_layout()
    #
    # output_image_path = result_path + "/classify.png"  # 定义保存路径和文件名
    # plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # # 非阻塞显示图形
    #
    # plt.show(block=False)
    #
    # # 暂停一段时间（例如 2 秒）
    # plt.pause(2)  # 2 秒后继续执行
    #
    # # 关闭图形窗口
    # plt.close()

    # ==================== ROC曲线和混淆矩阵 ====================
    # 创建二进制预测数组
    num_substances = y_test_classify_pred.shape[1]
    top2_indices = np.argsort(-y_test_classify_pred, axis=1)[:, :2]  # 按降序取前两个索引
    mask = np.zeros((y_test_classify_pred.shape[0], num_substances), dtype=bool)
    rows = np.arange(y_test_classify_pred.shape[0])[:, None]  # 生成样本行索引
    mask[rows, top2_indices] = True  # 标记前两个最高概率的位置
    y_pred_binary = mask.astype(int)  # 转换为二进制数组

    # 多标签分类转为单分类
    y_pred_binary_1d = tools.convert_to_1d(y_pred_binary, 5)
    y_test_1d = tools.convert_to_1d(y_test, 5)
    fig_path = result_path + '/cm_1d_test.png'
    report_path = result_path + '/cm_1d_test.txt'
    tools.plot_cm(y_test_1d, y_pred_binary_1d, data_vars, fig_path, report_path,
                  target_names=data_vars, normalize=False)

    # 多标签分类报告
    print("\n二维向量多标签 Classification Report:")
    print(classification_report(y_test, y_pred_binary,
                                target_names=['PC', 'PP', 'PS', 'PE', 'PVC'],
                                #target_names=[f"Substance_{i}" for i in range(args.num_substances)]
                                ))

    # ROC曲线和AUC值

    # 绘制ROC曲线
    output_image_path = result_path + "/roc_test.png"  # 定义保存路径和文件名
    output_image_path2 = result_path + "/roc_test2.png"  # 定义保存路径和文件名
    tools.plot_roc_curve(y_test, y_test_classify_pred, [f"Substance_{i}" for i in range(args.num_substances)],
                         output_image_path2)
    fpr, tpr, roc_auc = tools.plot_roc_curves2(y_test, y_test_classify_pred,
                                               [f"Substance_{i}" for i in range(args.num_substances)],
                                               output_image_path)
    fpr_output_name = result_path + '/fpr_test.xlsx'
    tpr_output_name = result_path + '/tpr_test.xlsx'
    roc_output_name = result_path + '/roc_test.xlsx'
    tools.dict_to_xlsx(fpr, fpr_output_name)
    tools.dict_to_xlsx(tpr, tpr_output_name)
    tools.dict_to_xlsx(roc_auc, roc_output_name)

    # 4. 混淆矩阵
    # 混淆矩阵可视化

    print("\nNormalized Confusion Matrices:")
    output_image_path = result_path + "/cm_test.png"  # 定义保存路径和文件名
    print(output_image_path)
    tools.plot_multilabel_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_classify_pred,
        output_image_path=output_image_path,
        # class_names=[f"Substance_{i}" for i in range(args.num_substances)],
        class_names=['PC', 'PP', 'PS', 'PE', 'PVC'],
        threshold=0.5,
        normalize=True
    )

    # # 5. 打印AUC值
    # print("\nAUC Scores:")
    # for i in range(args.num_substances):
    #     fpr, tpr, _ = roc_curve(y_test[:, i], y_test_classify_pred[:, i])
    #     roc_auc = auc(fpr, tpr)
    #     print(f"{args.MP_type[i]}: {roc_auc:.4f}")
    #
    # print('\nClassification_matrix:\n')
    # print(f'accuracy:{accuracy[-1]:.8f}')
    # print(f'precision:{precision[-1]:.8f}')
    # print(f'recall:{recall[-1]:.8f}')
    # print(f'val_accuracy:{val_accuracy[-1]:.8f}')
    # print(f'val_precision:{val_precision[-1]:.8f}')
    # print(f'val_recall:{val_recall[-1]:.8f}')

    # ====================fit 实测混合光谱结果评估 ====================
    spectra_mat = tools.load_mat_to_np(path_data_pred)

    data_vars_norm = [
        'PCPE_norm', 'PCPP_norm', 'PCPS_norm', 'PCPVC_norm',
        'PEPP_norm', 'PEPS_norm', 'PEPVC_norm',
        'PPPS_norm', 'PPPVC_norm',
        'PSPVC_norm'
    ]
    print("\n====== 实测混合光谱结果评估 ======")
    # 初始化数据列表
    spectra_mix_fit = []
    labels_mix_fit = []
    for var_name in data_vars_norm:
        parts_var_name = var_name.split("_")
        data = spectra_mat[var_name]
        print(var_name)
        # 解析标签（前两位和后两位物质）#######################################
        var_name = parts_var_name[0]
        substances = [var_name[:2], var_name[2:]]  # 假设变量名格式如 PCPE
        label = np.zeros(num_classes)
        for sub in substances:
            if sub in category_to_idx:
                label[category_to_idx[sub]] = 1

            # 添加到列表
        data_dict['X_extra'][var_name] = data
        data_dict['y_extra'][var_name].append(np.tile(label, (data.shape[0], 1)))  # 标签重复对应样本数
        spectra_mix_fit.append(data)
        labels_mix_fit.append(np.tile(label, (data.shape[0], 1)))  # 标签重复对应样本数

        print(label)
    print('Data have been loaded')
    # 合并所有数据
    samples = np.vstack(spectra_mix_fit)
    labels = np.vstack(labels_mix_fit)

    X_fit = np.expand_dims(samples, axis=-1)
    y_fit = labels
    # 绘制第一行的 X_test, y_test, y_test_pred 对比图
    # 模型拟合
    x_fit_A_pred, x_fit_B_pred, y_fit_classify_pred = model.predict(X_fit)

    df_X_fit = pd.DataFrame(X_fit.reshape(X_fit.shape[0], -1))
    df_x_fit_A_pred = pd.DataFrame(x_fit_A_pred.reshape(x_fit_A_pred.shape[0], -1))
    df_x_fit_B_pred = pd.DataFrame(x_fit_B_pred.reshape(x_fit_B_pred.shape[0], -1))

    # 保存为 CSV 文件
    df_X_fit.to_csv(result_path + '/X_extra.csv', index=False)
    df_x_fit_A_pred.to_csv(result_path + '/X_A_pred_fit.csv', index=False)
    df_x_fit_B_pred.to_csv(result_path + '/X_B_pred_fit.csv', index=False)

    # ==========================fit 每类混合物单独预测 ========================
    for var_name in data_vars:
        # 获取当前物质组合的测试数据
        X_fit_var = data_dict['X_extra'][var_name]

        # 调整数据形状以匹配模型输入 (添加通道维度)
        X_fit_var = np.expand_dims(X_fit_var, axis=-1)

        # 执行预测
        X1_fit_pred, X2_fit_pred, y_fit_pred = model.predict(X_fit_var)

        # 将预测结果存储到 data_dict 中
        data_dict['X1_fit_pred'][var_name] = X1_fit_pred
        data_dict['X2_fit_pred'][var_name] = X2_fit_pred
        data_dict['y_extra_pred'][var_name] = y_fit_pred

        # 打印进度
        print(f"Processed predictions for {var_name}")

    # 遍历所有物质组合
    var_to_labels = {
        "PCPE": ["PC", "PE"],
        "PCPP": ["PC", "PP"],
        "PCPS": ["PC", "PS"],
        "PCPVC": ["PC", "PVC"],
        "PEPP": ["PE", "PP"],
        "PEPS": ["PE", "PS"],
        "PEPVC": ["PE", "PVC"],
        "PPPS": ["PP", "PS"],
        "PPPVC": ["PP", "PVC"],
        "PSPVC": ["PS", "PVC"],
    }

    for var_name in data_dict['y_extra_pred']:
        # 获取真实标签和预测概率
        y_true = np.array(data_dict['y_extra'][var_name])  # 形状 (n_samples, n_classes)
        y_pred_proba = np.array(data_dict['y_extra_pred'][var_name])  # 形状 (n_samples, n_classes)
        y_true = np.squeeze(y_true)

        # 计算指标（使用Top-2策略）
        metrics = tools.evaluate_multilabel_accuracy(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            class_names=args.MP_type,  # 替换为你的类别名称列表
            var=var_name,
            var_to_labels=var_to_labels,
            type_mp=args.MP_type,
            top_k=2
        )

        # 将结果存入数据字典
        data_dict["metrics"][var_name] = metrics

        # 打印结果
        print(f"\n=== Metrics for {var_name} ===")
        print(f"Subset Accuracy （双标签同时正确）: {metrics['subset_accuracy']:.4f}")
        print(f"Average Sample Accuracy （单标签正确）: {metrics['average_sample_accuracy']:.4f}")
        for label, acc in metrics["label_accuracy"].items():
            print(f"Accuracy   {label}: {acc:.4f}")

    # ==================== fit ====================
    wavenumber = np.linspace(132.57, 4051.96, 1024)
    fig6 = plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.plot(wavenumber, X_fit[10], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的
    plt.plot(wavenumber, x_fit_A_pred[10], label='Spectrum A', linewidth=2, alpha=0.7)
    plt.plot(wavenumber, x_fit_B_pred[10], label='Spectrum B', linewidth=2, alpha=0.7)

    plt.title('Fit comparison')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(wavenumber, X_fit[10], label='Mixed spectra', linewidth=2, alpha=0.7)  #linewidth 线宽，alpha透明的

    plt.title('Mixture')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(wavenumber, x_fit_A_pred[10], label='Spectrum A', linewidth=2, alpha=0.7)

    plt.title('Fit Spectrum A')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(wavenumber, x_fit_B_pred[10], label='Spectrum B', linewidth=2, alpha=0.7)

    plt.title('Fit Spectrum B')
    plt.xlabel('Raman shift')
    plt.ylabel('Intensity')
    plt.legend()

    output_image_path = result_path + "/fit.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()

    # ==================== fit  classify ====================
    # ==================== fit  ROC曲线和混淆矩阵 ====================
    # y_fit_pred_binary = (y_fit_classify_pred > 0.5).astype(int)
    num_substances = y_fit_classify_pred.shape[1]
    top2_indices = np.argsort(-y_fit_classify_pred, axis=1)[:, :2]  # 按降序取前两个索引
    # 创建二进制预测数组
    mask = np.zeros((y_fit_classify_pred.shape[0], num_substances), dtype=bool)
    rows = np.arange(y_fit_classify_pred.shape[0])[:, None]  # 生成样本行索引
    mask[rows, top2_indices] = True  # 标记前两个最高概率的位置
    y_fit_pred_binary = mask.astype(int)  # 转换为二进制数组

    # 多标签分类转为单分类
    y_fit_pred_binary_1d = tools.convert_to_1d(y_fit_pred_binary, 5)
    y_fit_1d = tools.convert_to_1d(y_fit, 5)
    fig_path = result_path + '/cm_1d_fit.png'
    report_path = result_path + '/cm_1d_fit.txt'
    tools.plot_cm(y_fit_1d, y_fit_pred_binary_1d, data_vars, fig_path, report_path,
                  target_names=data_vars, normalize=False)

    # 多标签分类报告
    print("\n二维向量多标签Classification Report:")
    print(classification_report(y_fit, y_fit_pred_binary,digits=4,
                                target_names=['PC', 'PP', 'PS', 'PE', 'PVC'],
                                #target_names=[f"Substance_{i}" for i in range(args.num_substances)]
                                ))

    # 绘制ROC曲线
    output_image_path = result_path + "/roc_fit.png"  # 定义保存路径和文件名
    fpr, tpr, roc_auc = tools.plot_roc_curves2(y_fit, y_fit_classify_pred,
                                               [f"Substance_{i}" for i in range(args.num_substances)],
                                               output_image_path=output_image_path)
    fpr_output_name = result_path + '/fpr_fit.xlsx'
    tpr_output_name = result_path + '/tpr_fit.xlsx'
    roc_output_name = result_path + '/roc_fit.xlsx'
    tools.dict_to_xlsx(fpr, fpr_output_name)
    tools.dict_to_xlsx(tpr, tpr_output_name)
    tools.dict_to_xlsx(roc_auc, roc_output_name)

    # 混淆矩阵可视化
    # 可选：添加归一化版本的混淆矩阵
    print("\nNormalized Confusion Matrices:")
    output_image_path = result_path + "/cm_fit.png"  # 定义保存路径和文件名
    print(output_image_path)
    tools.plot_multilabel_confusion_matrix(
        y_true=y_fit,
        y_pred=y_fit_classify_pred,
        output_image_path=output_image_path,
        #class_names=[f"Substance_{i}" for i in range(args.num_substances)],
        class_names=['PC', 'PP', 'PS', 'PE', 'PVC'],
        threshold=0.5,
        normalize=True
    )

    # 5. 打印AUC值
    print("\nAUC Scores:")
    for i in range(args.num_substances):
        fpr, tpr, _ = roc_curve(y_fit[:, i], y_fit_classify_pred[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"{args.MP_type[i]}: {roc_auc:.4f}")

    #  ========== Grad-CAM ======================================
    heatmaps_dict = {
        'input': {m: [] for m in data_vars},
        'heatmap1': {m: [] for m in data_vars},  # 改为两个独立键
        'heatmap2': {m: [] for m in data_vars},
        'top_classes': {m: [] for m in data_vars}
    }
    index = 1
    for var in data_vars:
        X_test_cam = data_dict['X_test'][var][0]
        X_test_cam = np.squeeze(X_test_cam)
        X_test_cam = X_test_cam.reshape(1, 1024, 1)
        heatmaps_dict['input'][var] = data_dict['X_test'][var][0]

        # 生成热力图
        heatmaps, top_classes = tools.generate_gradcam_heatmap(model, X_test_cam, top_k=2)

        # 分别存储前两个热力图
        if len(top_classes) >= 1:
            heatmaps_dict['heatmap1'][var] = heatmaps[top_classes[0]]  # 最高概率类别
        if len(top_classes) >= 2:
            heatmaps_dict['heatmap2'][var] = heatmaps[top_classes[1]]  # 次高概率类别
        heatmaps_dict['top_classes'][var] = top_classes

    # 步骤8：可视化
    plt.figure(figsize=(12, 6))
    # 热力图叠加
    heatmaps_names = [
        'PCPE', 'PCPP', 'PCPS', 'PCPVC',
        'PEPP', 'PEPS', 'PEPVC', 'PPPS',
        'PPPVC', 'PSPVC'
    ]
    i = 1
    for var in data_vars:
        # 计算子图位置和对应的系数
        subplot_pos = (2, 5, i)

        # 创建子图
        plt.subplot(*subplot_pos)

        x_data = heatmaps_dict['input'][var]
        # 绘制输入光谱
        plt.plot(x_data, color='gray', alpha=0.3, label='Input')
        # 绘制热力图
        heatmap1 = heatmaps_dict['heatmap1'][var]
        heatmap2 = heatmaps_dict['heatmap2'][var]
        plt.plot(heatmap1, label=f"heatmap1")
        plt.plot(heatmap2, label=f"heatmap2")
        # 设置图表属性
        plt.title(var)
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')
        plt.legend()
        i = i + 1
    plt.tight_layout()
    output_image_path = result_path + "/gradcam.png"  # 定义保存路径和文件名
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    # 非阻塞显示图形
    plt.show(block=False)

    # 暂停一段时间（例如 2 秒）
    plt.pause(2)  # 2 秒后继续执行

    # 关闭图形窗口
    plt.close()
    print(f"\nGradCAM successfully saved to: ", output_image_path)

    input_df = pd.DataFrame(heatmaps_dict['input'])
    input_df.to_csv(result_path + "\grad cam_input_data.csv", index=False)

    # 保存 heatmaps 数据
    heatmaps_df = pd.DataFrame(heatmaps_dict['heatmap1'])
    heatmaps_df.to_csv(result_path + "\grad cam_heatmap1_data.csv", index=False)
    # 保存 heatmaps 数据
    heatmaps_df = pd.DataFrame(heatmaps_dict['heatmap2'])
    heatmaps_df.to_csv(result_path + "\grad cam_heatmap2_data.csv", index=False)

    # 保存 top_classes 数据
    top_classes_df = pd.DataFrame(heatmaps_dict['top_classes'])
    top_classes_df.to_csv(result_path + "\grad cam_top_classes.csv", index=False)

    # ===================  保存到mat ===================
    containers_matrix = {
        'simulate_raw': X,
        'simulate_spectrum_1_target': X1,
        'simulate_spectrum_2_target': X2,
        'simulate_spectrum_1_pred': x1_pred,
        'simulate_spectrum_2_pred': x2_pred,

        'simulate_raw_test': X_test,
        'simulate_spectrum_1_target_test': X1_test,
        'simulate_spectrum_2_target_test': X2_test,
        'simulate_spectrum_1_pred_test': x_test_A_pred,
        'simulate_spectrum_2_pred_test': x_test_B_pred,

        'real_raw': X_fit,
        'real_spectrum_1': x_fit_A_pred,
        'real_spectrum_2': x_fit_B_pred

    }

    out_path = result_path+'/' + current_time + '_' + 'unmixing_matrix.mat'
    save_dict = {
        **containers_matrix,
    }
    savemat(
        out_path,
        save_dict,
        oned_as='column'  # 确保一维数组保存为列向量
    )
    print(f"Data successfully saved to: ", out_path)
