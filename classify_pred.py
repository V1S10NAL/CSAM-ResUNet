"""
用于拟合额外其他光谱
"""
import math
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
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--dataset', type=str, default='test_data', help='dataset')  # 此处选择数据集
parser.add_argument('--num_substances', type=int, default=5, help='num_substances')
parser.add_argument('--MP_type', type=str, default=['PC', 'PE', 'PP', 'PS', 'PVC'], help='microplastics types')
parser.add_argument('--save2mat', type=bool, default=True, help='save2mat')

args = parser.parse_args(args=[])

# 设置随机种子
seed = 1958
save2mat = True
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

path_list_nn = [
    '.\\dataset\\dataset_mixture_denoised\\dataset1_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset2_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset3_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset4_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset5_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset6_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset7_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset8_denoised.mat',
    '.\\dataset\\dataset_mixture_denoised\\dataset9_denoised.mat',
]

path_list_wtd = [

]

path_list_sg = [

]

path_list = path_list_nn

for path in path_list:

    print(f"\nLoaded {path}")
    path_data_pred = path
    run_path: str = 'CSAM_ResUNet_classification_20250617_085150'                          #################CSAM_ResUNet_classification_20250524_135115###CSAM_ResUNet_classification_20250524_154531######改模型 CSAM_ResUNet_classification

    parts = run_path.split('_')
    if len(parts) >= 2:
        net_name = '_'.join(parts[:-2])  # 去除最后两个分段
    else:
        net_name = run_path  # 没有足够分段时保留原字符串
    dataset = run_path


    start_time = time.perf_counter()
    spectra_mat = tools.load_mat_to_np(path_data_pred)  ###dataset_train_mix_1000.mat     dataset_train_mixed_mp.mat

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
    # for var_name in data_vars:
    #     data = spectra_mat[var_name]
    #     print(var_name)
    #     # 解析标签（前两位和后两位物质）#######################################
    #     substances = [var_name[:2], var_name[2:]]  # 假设变量名格式如 PCPE
    #     label = np.zeros(num_classes)
    #     for sub in substances:
    #         if sub in category_to_idx:
    #             label[category_to_idx[sub]] = 1
    #
    #     data_dict['X'][var_name] = data
    #     data_dict['y'][var_name].append(np.tile(label, (data.shape[0], 1)))  # 标签重复对应样本数
    #     # 提取端元#################################################
    #     mp1 = var_name + '_' + var_name[:2]
    #     data_mp1 = spectra_mat[mp1]
    #     data_dict['X1'][var_name].append(data_mp1)
    #     mp2 = var_name + '_' + var_name[2:]
    #     data_mp2 = spectra_mat[mp2]
    #     data_dict['X2'][var_name].append(data_mp2)
    #     print(label)
    #
    #     num_spectra = num_spectra + data.shape[0]
    #
    #     (data_dict['X_train'][var_name],
    #      data_dict['X_temp'][var_name],
    #      data_dict['X1_train'][var_name],
    #      data_dict['X1_temp'][var_name],
    #      data_dict['X2_train'][var_name],
    #      data_dict['X2_temp'][var_name],
    #      data_dict['y_train'][var_name],
    #      data_dict['y_temp'][var_name]) = train_test_split(
    #         np.squeeze(data_dict['X'][var_name]),
    #         np.squeeze(data_dict['X1'][var_name]),
    #         np.squeeze(data_dict['X2'][var_name]),
    #         np.squeeze(data_dict['y'][var_name]), test_size=0.3, random_state=seed)
    #
    #     (data_dict['X_val'][var_name],
    #      data_dict['X_test'][var_name],
    #      data_dict['X1_val'][var_name],
    #      data_dict['X1_test'][var_name],
    #      data_dict['X2_val'][var_name],
    #      data_dict['X2_test'][var_name],
    #      data_dict['y_val'][var_name],
    #      data_dict['y_test'][var_name]) = train_test_split(
    #         np.squeeze(data_dict['X_temp'][var_name]),
    #         np.squeeze(data_dict['X1_temp'][var_name]),
    #         np.squeeze(data_dict['X2_temp'][var_name]),
    #         np.squeeze(data_dict['y_temp'][var_name]), test_size=1 / 3, random_state=seed)
    #
    #     X.append(np.squeeze(data_dict['X'][var_name]))
    #     X1.append(np.squeeze(data_dict['X1'][var_name]))
    #     X2.append(np.squeeze(data_dict['X2'][var_name]))
    #     y.append(np.squeeze(data_dict['y'][var_name]))
    #
    #     X_train.append(np.squeeze(data_dict['X_train'][var_name]))
    #     X1_train.append(np.squeeze(data_dict['X1_train'][var_name]))
    #     X2_train.append(np.squeeze(data_dict['X2_train'][var_name]))
    #     y_train.append(np.squeeze(data_dict['y_train'][var_name]))
    #
    #     X_val.append(np.squeeze(data_dict['X_val'][var_name]))
    #     X1_val.append(np.squeeze(data_dict['X1_val'][var_name]))
    #     X2_val.append(np.squeeze(data_dict['X2_val'][var_name]))
    #     y_val.append(np.squeeze(data_dict['y_val'][var_name]))
    #
    #     X_test.append(np.squeeze(data_dict['X_test'][var_name]))
    #     X1_test.append(np.squeeze(data_dict['X1_test'][var_name]))
    #     X2_test.append(np.squeeze(data_dict['X2_test'][var_name]))
    #     y_test.append(np.squeeze(data_dict['y_test'][var_name]))

    # X = np.vstack(X)
    # X1 = np.vstack(X1)
    # X2 = np.vstack(X2)
    # y = np.vstack(y)
    #
    # X_train = np.vstack(X_train)
    # X1_train = np.vstack(X1_train)
    # X2_train = np.vstack(X2_train)
    # y_train = np.vstack(y_train)
    #
    # X_val = np.vstack(X_val)
    # X1_val = np.vstack(X1_val)
    # X2_val = np.vstack(X2_val)
    # y_val = np.vstack(y_val)
    #
    # X_test = np.vstack(X_test)
    # X1_test = np.vstack(X1_test)
    # X2_test = np.vstack(X2_test)
    # y_test = np.vstack(y_test)
    # # 合并所有数据


    end_time = time.perf_counter()
    loading_time = end_time - start_time
    print(f"loading time:{loading_time:.6f}")


    # num_spectra = X.shape[0]                                       # 光谱数

    # args, unknown = parser.parse_known_args()
    # parser.add_argument('--num_spectra', type=int, default=num_spectra, help="number of spectra")
    # args = parser.parse_args()

    # X_extra = np.expand_dims(X, axis=-1)
    #X1_extra = np.expand_dims(X1, axis=-1)
    #X2_extra = np.expand_dims(X2, axis=-1)
    # y_extra= np.expand_dims(y, axis=-1)

    # X_reality= simulated_spectra
    # y_extra= original_spectra
    # 载入训练好的模型
    # 修改路径选择训练好的模型


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
    #
    # x_extra_A_pred, x_extra_B_pred, y_extra_pred = model.predict(X_extra)


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
            threshold= 0.5,
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
    fig = plt.figure(figsize=(6, 6))
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
    # plt.show(block=False)
    # plt.pause(1)  # 2 秒后继续执行
    #
    # # 关闭图形窗口
    # plt.close(fig)

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

    print("数据集路径：")
    print(path_data_pred)
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

    if save2mat:
        # ===================  保存到mat ===================
        containers_matrix = {
            'real_raw': X_fit,
            'real_spectrum_1_pred': x_fit_A_pred,
            'real_spectrum_2_pred': x_fit_B_pred
        }
        out_path = result_path + '/' + current_time + '_' + 'unmixing_matrix.mat'
        save_dict = {
            **containers_matrix,
        }
        savemat(
            out_path,
            save_dict,
            oned_as='column'  # 确保一维数组保存为列向量
        )
        print(f"Data successfully saved to: ", out_path)
