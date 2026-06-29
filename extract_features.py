# -*- coding: utf-8 -*-
"""
特征提取模块 —— HOG（Histogram of Oriented Gradients，梯度方向直方图）特征提取
=================================================================================
核心概念：
    1. HOG 特征的基本思想：
       一幅图像中，物体的形状和外观可以通过局部梯度（边缘）的方向分布来描述。
       梯度的方向表示了边缘的朝向，梯度的幅度表示了边缘的强度。

    2. HOG 计算流程（分步详解）：
       Step 1 - 计算梯度:  用 Sobel 算子 [-1,0,1] 计算每个像素的水平梯度 Gx 和垂直梯度 Gy
                          梯度幅度 = sqrt(Gx² + Gy²)
                          梯度方向 = arctan(Gy / Gx)，映射到 [0°, 180°)
       Step 2 - Cell 统计: 将图像划分为若干 cell（如 8×8 像素），在每个 cell 内
                          统计梯度方向直方图（9 个 bin，每个 bin 覆盖 20°）
       Step 3 - Block 归一化: 将相邻的 2×2 个 cell 组成一个 block，对 block 内的
                          所有直方图做 L2 归一化（减少光照影响）
       Step 4 - 拼接:      将所有 block 的归一化直方图拼接成最终的特征向量

    3. 特征向量长度计算公式：
       设图像尺寸为 H×W，cell 大小为 (ch, cw)，block 大小为 (bh, bw)，n_bins=9

       cell 数量:        n_cells_y = H / ch, n_cells_x = W / cw
       block 数量:       n_blocks_y = n_cells_y - bh + 1
                        n_blocks_x = n_cells_x - bw + 1
       每个 block 含:    bh × bw × n_bins 维
       总特征维度:       n_blocks_y × n_blocks_x × bh × bw × n_bins

       ★ 对于 64×64 图像，skimage hog 默认参数 (cell=8×8, block=3×3, bins=9):
         n_cells = 64/8 = 8×8 = 64 个 cell
         n_blocks = (8-3+1) × (8-3+1) = 6×6 = 36 个 block
         每个 block: 3×3×9 = 81 维
         总维度 = 36 × 81 = 2916 维

    4. 为什么 HOG 适合手势识别？
       - 手势的边缘和轮廓方向是区分不同手势的关键线索
       - HOG 描述的是局部形状信息，对小的形变和光照变化不敏感
       - 与像素值直接分类相比，HOG 捕捉了高层次的结构信息

参考文献：
    Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection.
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import exposure

# ==================== 配置参数 ====================
# 输入目录（预处理后的图像）
PROCESSED_DATA_DIR = 'processed_data'
# 输出目录（特征文件）
FEATURES_DIR = 'processed_data'

# HOG 参数（skimage 默认值）
HOG_ORIENTATIONS = 9       # 方向 bin 数量（9 个 bin，每个 20°）
HOG_PIXELS_PER_CELL = (8, 8)    # 每个 cell 的像素尺寸
HOG_CELLS_PER_BLOCK = (3, 3)    # 每个 block 包含的 cell 数量
# 注：使用 (3,3) 的 block 而非常见的 (2,2)，是因为 64×64 图像较小，
# (3,3) block 能提供更大的归一化窗口，有利于捕捉手势的整体形状


def extract_hog_single(image):
    """
    【核心函数1】对单张 64×64 灰度图提取 HOG 特征
    ===============================================
    使用 skimage.feature.hog 函数，该函数内部完成了：
      1. 计算梯度（幅度和方向）
      2. 在每个 cell 内统计梯度方向直方图
      3. block 级别的 L2 归一化
      4. 拼接成最终特征向量

    :param image: 64×64 的灰度图像（uint8, 0-255）
    :return: features —— 一维 numpy 数组，长度为 2916
    """
    # 调用 skimage 的 hog 函数
    # 参数说明：
    #   orientations=9:        将 180° 分为 9 个 bin，每个 bin 20°
    #   pixels_per_cell=(8,8): 每个 cell 为 8×8 像素
    #   cells_per_block=(3,3): 每个 block 包含 3×3 个 cell
    #   visualize=False:       不返回 HOG 可视化图像
    #   channel_axis=None:     输入是单通道灰度图（非多通道）
    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        channel_axis=None
    )
    return features


def extract_hog_with_visualization(image):
    """
    【可视化函数】提取 HOG 特征并返回可视化图像
    ============================================
    用于报告中的 HOG 特征示意图。
    可视化图像中，每个 cell 用星形线表示其梯度方向直方图——
    线条越亮/越长表示该方向的梯度越强。

    :param image: 64×64 灰度图
    :return: (features, hog_image) —— 特征向量和 HOG 可视化图
    """
    features, hog_image = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=True,
        channel_axis=None
    )
    # 对 HOG 可视化图像做直方图均衡化，使线条更清晰可见
    hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
    hog_image = hog_image.astype(np.uint8)

    return features, hog_image


def augment_image(img):
    """
    【数据增强】对单张 64×64 灰度图生成多个增强版本
    ==================================================
    增强策略（模拟真实场景中的变化）：
      1. 水平翻转 —— 模拟左右手或不同方向
      2. 小角度旋转 ±10° —— 模拟手部倾斜
      3. 亮度变化 ±20 —— 模拟不同光照条件

    每张原图生成 6 个增强版本（含原图）。

    :param img: 64×64 灰度图
    :return: 增强后的图像列表
    """
    augmented = [img]  # 原图始终保留

    h, w = img.shape[:2]

    # 1. 水平翻转（模拟左右手互换）
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)

    # 2. 小角度旋转 ±10°（模拟手部轻微倾斜）
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
        augmented.append(rotated)

    # 3. 亮度调整（模拟不同光照）
    for delta in [-25, 25]:
        bright = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        augmented.append(bright)

    return augmented


def extract_all_features(save=True):
    """
    【批量处理】对所有预处理后的图像提取 HOG 特征
    ===============================================
    遍历 processed_data/ 下每个类别的所有图像，逐一提取 HOG 特征，
    并将特征矩阵和标签向量保存为 .npy 文件。

    :param save: 是否保存到文件
    :return: X (特征矩阵), y (标签向量)
    """
    print('=' * 55)
    print('  HOG 特征提取')
    print('=' * 55)

    X = []   # 特征矩阵（每行是一个样本的特征向量）
    y = []   # 标签向量（每个样本对应的类别编号）

    # 遍历每个手势类别
    for class_name in sorted(os.listdir(PROCESSED_DATA_DIR)):
        class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 从文件夹名中提取标签（如 "0_fist" → 0, "1_palm" → 1）
        try:
            label = int(class_name.split('_')[0])
        except ValueError:
            print(f'  [警告] 无法解析标签: {class_name}，跳过')
            continue

        # 获取所有图像（排序确保与 train_svm.py 中顺序一致）
        image_files = sorted([f for f in os.listdir(class_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

        if len(image_files) == 0:
            print(f'  [{class_name}] 无图像，跳过')
            continue

        print(f'\n  [{class_name}] label={label}, {len(image_files)} 张图像')

        for fname in image_files:
            img_path = os.path.join(class_dir, fname)
            # 读取预处理后的灰度图（此时已是 64×64）
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 提取 HOG 特征（数据增强移至 train_svm.py 以避免数据泄露）
            features = extract_hog_single(img)
            X.append(features)
            y.append(label)

    # 转为 numpy 数组
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f'\n{"="*55}')
    print(f'  特征矩阵 X: {X.shape}  (样本数 × 特征维度)')
    print(f'  标签向量 y: {y.shape}  (样本数,)')
    print(f'  每个样本的特征维度: {X.shape[1]} 维')
    print(f'  各类别样本数:')
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f'    类别 {lbl}: {cnt} 个样本')
    print(f'{"="*55}')

    if save:
        # 保存特征矩阵和标签为 .npy 文件
        np.save(os.path.join(FEATURES_DIR, 'features_hog.npy'), X)
        np.save(os.path.join(FEATURES_DIR, 'labels.npy'), y)
        print(f'  特征已保存到: {FEATURES_DIR}/features_hog.npy')
        print(f'  标签已保存到: {FEATURES_DIR}/labels.npy')

    return X, y


def demo_hog_visualization(image_path):
    """
    【演示函数】展示单张图像的 HOG 特征可视化
    用于调试和报告插图
    """
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'无法读取: {image_path}')
        return

    # 缩放到 64×64
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

    # 提取 HOG 特征和可视化图
    features, hog_img = extract_hog_with_visualization(img)

    # 显示
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original (64×64)')
    axes[0].axis('off')

    axes[1].imshow(hog_img, cmap='gray')
    axes[1].set_title(f'HOG Visualization\nFeature dim: {len(features)}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # 单张可视化模式
        demo_hog_visualization(sys.argv[1])
    else:
        # 批量提取模式
        extract_all_features(save=True)
