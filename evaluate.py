# -*- coding: utf-8 -*-
"""
模型评估模块 —— 准确率、混淆矩阵、错误案例分析
===================================================
核心概念：
    1. 混淆矩阵（Confusion Matrix）：
       - 行 (row)    = 真实标签 (True Label)
       - 列 (column) = 预测标签 (Predicted Label)
       - 对角线元素 = 正确分类的样本数
       - 非对角线   = 被错误分类到其他类别的样本数

       示例（3类）：          预测
                       猫    狗    鸟
             真实 猫  [ 18    2    0 ]  ← 18只猫被正确识别，2只被误认为狗
                  狗  [  1   17    2 ]
                  鸟  [  0    3   17 ]

    2. 从混淆矩阵可以推导：
       - Accuracy  = 对角线之和 / 全部样本数
       - Precision = 对角线 / 列和
       - Recall    = 对角线 / 行和

    3. 错误分析的意义：
       观察混淆矩阵中最容易混淆的类别对，思考：
       - 这两类手势在视觉上是否相似？（如 OK 手势和握拳）
       - 特征提取是否充分捕捉了区分性信息？
       - 训练数据是否足够多样？
"""

import numpy as np
import os
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
import pickle
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display

# ==================== 配置参数 ====================
FEATURES_DIR = 'processed_data'
MODEL_DIR = 'models'
RESULT_DIR = 'results'

# Gesture class names (English for font compatibility)
CLASS_NAMES_CN = ['fist', 'palm', 'thumb', 'ok', 'victory']
CLASS_NAMES_EN = ['fist', 'palm', 'thumb', 'ok', 'victory']

# Try to configure Chinese font support (fallback to English if unavailable)
try:
    import matplotlib.font_manager as fm
    # Try common Chinese fonts on Windows
    for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            # Reload seaborn with new font
            import importlib
            importlib.reload(sns)
            break
        except Exception:
            continue
except Exception:
    pass


def load_results():
    """加载保存的测试预测结果"""
    y_test = np.load(os.path.join(FEATURES_DIR, 'y_test.npy'))
    y_pred = np.load(os.path.join(FEATURES_DIR, 'y_pred.npy'))
    return y_test, y_pred


def plot_confusion_matrix(y_true, y_pred, save=True, show=True):
    """
    【核心函数1】绘制混淆矩阵 —— 同时显示数字和百分比
    ====================================================
    使用 matplotlib + seaborn 绘制热力图风格的混淆矩阵

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param save:   是否保存为图片
    :param show:   是否显示
    """
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 2. 计算百分比混淆矩阵（每行 = 每个真实类别的分布）
    #    行归一化：每行的值 / 该行总和 → 显示该类被分到各个类别的比例
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # 3. 创建标注文本（同时显示数字和百分比）
    n_classes = cm.shape[0]
    annot = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            else:
                annot[i, j] = '0\n(0.0%)'

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用 seaborn 的热力图
    sns.heatmap(
        cm_percent,                      # 用百分比数据决定颜色深浅
        annot=annot,                     # 显示的文字（数字+百分比）
        fmt='',                          # 因为 annot 已经是字符串，不需要格式化
        cmap='Blues',                    # 蓝色渐变配色
        xticklabels=CLASS_NAMES_CN,
        yticklabels=CLASS_NAMES_CN,
        vmin=0, vmax=100,
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=1,                    # 格子之间的白线宽度
        linecolor='white',
        ax=ax,
    )

    ax.set_xlabel('Predicted Label (预测标签)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label (真实标签)', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix — Hand Gesture Recognition\n数字=样本数, 括号内=行百分比',
                 fontsize=14, fontweight='bold', pad=20)

    # 将对角线单元格加粗边框标记
    for i in range(n_classes):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                   fill=False, edgecolor='red',
                                   lw=3, linestyle='-'))

    plt.tight_layout()

    if save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'),
                    dpi=200, bbox_inches='tight')
        print(f'混淆矩阵已保存到: {RESULT_DIR}/confusion_matrix.png')

    if show:
        plt.show()
    else:
        plt.close()


def visualize_misclassified(y_true, y_pred, class_names=None, save=True, show=True):
    """
    【新增函数】可视化被错误分类的样本
    ====================================
    从 processed_data/ 中加载图像，展示被分错的样本，
    并显示真实标签 vs 预测标签。
    这比只看数字更能帮助理解模型为什么会犯错。

    输出：一张拼接图，包含前 10 个错误样本（如果够10个的话）
    """
    if class_names is None:
        class_names = CLASS_NAMES_EN

    # 找出所有错误分类的索引
    error_indices = np.where(y_true != y_pred)[0]
    if len(error_indices) == 0:
        print('  没有错误样本，跳过错判可视化。')
        return

    # 从 processed_data 加载对应的图像
    # 先收集所有图像路径
    all_image_paths = []
    all_image_labels = []
    proc_dir = FEATURES_DIR  # 'processed_data'
    for class_folder in sorted(os.listdir(proc_dir)):
        class_path = os.path.join(proc_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        try:
            lbl = int(class_folder.split('_')[0])
        except ValueError:
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                all_image_paths.append(os.path.join(class_path, fname))
                all_image_labels.append(lbl)

    # 确保数量一致
    if len(all_image_labels) != len(y_true):
        print(f'  [提示] 图像数量({len(all_image_labels)})与标签数({len(y_true)})不匹配，跳过错判可视化。')
        return

    # 取前 10 个错误样本
    n_show = min(10, len(error_indices))
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    for i in range(n_show):
        idx = error_indices[i]
        ax = axes[i]
        # 读取对应的图像
        try:
            img = cv2.imread(all_image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ax.imshow(img, cmap='gray')
            else:
                ax.text(0.5, 0.5, 'Image not found', transform=ax.transAxes, ha='center')
        except Exception:
            ax.text(0.5, 0.5, 'Load error', transform=ax.transAxes, ha='center')

        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        # 红色表示错误
        ax.set_title(f'True: {true_name}\nPred: {pred_name}',
                    fontsize=10, color='red', fontweight='bold')
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_show, 10):
        axes[i].axis('off')

    fig.suptitle('Misclassified Samples (Red = Wrong Prediction)',
                 fontsize=14, fontweight='bold', color='darkred')

    if save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULT_DIR, 'misclassified_samples.png'),
                    dpi=200, bbox_inches='tight')
        print(f'错判样本图已保存到: {RESULT_DIR}/misclassified_samples.png')

    if show:
        plt.show()
    else:
        plt.close()


def analyze_errors(y_true, y_pred, X_test, class_names=None):
    """
    【核心函数2】错误案例分析
    ============================
    找出测试集中分类错误的样本，分析可能的原因。
    输出：
      - 每个错误样本的信息（真实标签、预测标签、置信度）
      - 最容易混淆的类别对统计

    这一部分是报告中"错误分析"的核心内容。
    """
    if class_names is None:
        class_names = CLASS_NAMES_EN

    # 1. 找出所有错误分类的索引
    errors = np.where(y_true != y_pred)[0]

    print(f'\n{"="*55}')
    print(f'  错误案例分析')
    print(f'{"="*55}')
    print(f'  总测试样本: {len(y_true)}')
    print(f'  错误样本: {len(errors)} ({len(errors)/len(y_true)*100:.1f}%)')

    if len(errors) == 0:
        print('  没有错误样本，无需分析。')
        return

    # 2. 统计混淆对（哪两类之间最容易混淆）
    confusion_pairs = {}
    for idx in errors:
        pair = (y_true[idx], y_pred[idx])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    print('\n最容易混淆的类别对（真实→预测）：')
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true_lbl, pred_lbl), count in sorted_pairs[:5]:
        print(f'  {class_names[true_lbl]:10s} → {class_names[pred_lbl]:10s}: '
              f'{count} 次')

    # 3. 输出前几个错误样本的详细信息
    print(f'\n前 10 个错误样本详情：')
    print(f'{"序号":<6} {"真实标签":<16} {"预测标签":<16}')
    print('-' * 40)
    for i, idx in enumerate(errors[:10]):
        true_name = f'{y_true[idx]}-{class_names[y_true[idx]]}'
        pred_name = f'{y_pred[idx]}-{class_names[y_pred[idx]]}'
        print(f'{i+1:<6} {true_name:<16} {pred_name:<16}')

    return errors


def plot_per_class_accuracy(y_true, y_pred, class_names=None, save=True, show=True):
    """
    【可视化函数】绘制每个类别的准确率柱状图
    """
    if class_names is None:
        class_names = CLASS_NAMES_EN

    cm = confusion_matrix(y_true, y_pred)
    # 每个类别的准确率 = 该类被正确分类的样本 / 该类的总样本
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(range(len(class_names)))
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, edgecolor='black')

    # 在柱子上标注数值
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc*100:.1f}%', ha='center', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(RESULT_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULT_DIR, 'per_class_accuracy.png'),
                    dpi=200, bbox_inches='tight')
        print(f'各类别准确率已保存到: {RESULT_DIR}/per_class_accuracy.png')

    if show:
        plt.show()
    else:
        plt.close()


def generate_report(y_true, y_pred):
    """
    【报告生成】输出完整的评估报告
    """
    print('\n' + '=' * 55)
    print('  完整评估报告')
    print('=' * 55)

    # 1. 整体准确率
    acc = accuracy_score(y_true, y_pred)
    print(f'\n整体准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)')

    # 2. 混淆矩阵（数值形式）
    cm = confusion_matrix(y_true, y_pred)
    print(f'\n混淆矩阵 (行=真实, 列=预测):')
    header = '          ' + '  '.join([f'{n:>8s}' for n in CLASS_NAMES_EN])
    print(header)
    for i, name in enumerate(CLASS_NAMES_EN):
        row = '  '.join([f'{cm[i,j]:8d}' for j in range(cm.shape[1])])
        print(f'{name:>8s}  {row}')

    # 3. 分类报告
    print(f'\n详细分类报告：')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES_EN, digits=4))

    return acc, cm


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1. 加载预测结果
    try:
        y_test, y_pred = load_results()
    except FileNotFoundError:
        print('[提示] 未找到预存的预测结果，将使用模型重新预测...')
        # 加载模型和测试数据
        X_test = np.load(os.path.join(FEATURES_DIR, 'features_hog.npy'))
        y_test = np.load(os.path.join(FEATURES_DIR, 'labels.npy'))

        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X_test, y_test, test_size=0.3, random_state=42, stratify=y_test
        )

        with open(os.path.join(MODEL_DIR, 'svm_gesture.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

    # 2. 生成完整报告
    acc, cm = generate_report(y_test, y_pred)

    # 3. 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, save=True, show=True)

    # 4. 绘制各类别准确率
    plot_per_class_accuracy(y_test, y_pred, save=True, show=True)

    # 5. 错误案例分析
    # 尝试加载特征用于错误分析（可选）
    try:
        X_test_feat = np.load(os.path.join(FEATURES_DIR, 'features_hog.npy'))
    except FileNotFoundError:
        X_test_feat = None
    errors = analyze_errors(y_test, y_pred, X_test_feat)

    # 6. 错判样本可视化（新增）
    visualize_misclassified(y_test, y_pred, save=True, show=True)

    print(f'\n所有评估结果已保存到: {RESULT_DIR}/')
    return acc, cm, errors


if __name__ == '__main__':
    main()
