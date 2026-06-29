# -*- coding: utf-8 -*-
"""
SVM 分类器训练模块 —— 基于 HOG 特征的手势多分类
===================================================
核心概念：
    1. SVM（Support Vector Machine，支持向量机）：
       - 目标：在特征空间中找到一个最优超平面，将不同类别的样本分开
       - "最大间隔"原则：选择使得距离超平面最近的样本点（支持向量）到超平面
         的距离最大化的超平面 → 泛化能力最强

    2. 核函数（Kernel Function）：
       当数据在原始空间中线性不可分时，通过核函数将数据映射到高维空间，
       使数据在高维空间中线性可分。

       常用核函数：
       - linear:  线性核，K(x,y) = x·y
                  参数少、速度快、适合高维稀疏特征（如文本）
       - rbf:     高斯径向基核，K(x,y) = exp(-γ||x-y||²)
                  最常用的非线性核，适合大多数中等规模数据
       - poly:    多项式核，K(x,y) = (γ x·y + r)^d

    3. 关键超参数：
       - C（惩罚系数）: 权衡训练误差与模型复杂度
           C 大 → 更关注正确分类每个训练样本 → 容易过拟合
           C 小 → 允许更多分类错误 → 间隔更大 → 泛化更好
       - gamma（RBF 核参数）: 控制单个样本的影响范围
           gamma 大 → 影响范围小 → 决策边界更复杂 → 容易过拟合
           gamma 小 → 影响范围大 → 决策边界更平滑 → 可能欠拟合

    4. 多分类策略 —— One-vs-One (OvO)：
       sklearn 的 SVC 默认使用 OvO 策略：
       - 对 K 个类别，两两训练 K(K-1)/2 个二分类器
       - 预测时：每个分类器投票，得票最多的类别为最终结果
       - 优点：每个二分类器只需处理两类，训练更快

    5. 为什么 HOG + SVM 适合手势识别？
       - HOG 提取了形状/边缘的结构化特征（高维稀疏向量）
       - SVM 在高维空间表现优异，且泛化能力强
       - 组合在小样本场景下效果尤其好
"""

import numpy as np
import os
import pickle
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ==================== 配置参数 ====================
# 数据路径
FEATURES_DIR = 'processed_data'
FEATURES_FILE = os.path.join(FEATURES_DIR, 'features_hog.npy')
LABELS_FILE = os.path.join(FEATURES_DIR, 'labels.npy')

# 模型保存路径
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'svm_gesture.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')

# SVM 超参数（初始值，后续可通过网格搜索优化）
SVM_KERNEL = 'rbf'         # 核函数类型
SVM_C = 10.0               # 惩罚系数 C
SVM_GAMMA = 'scale'        # RBF 核的 gamma 参数（'scale' = 1/(n_features * X.var())）
SVM_RANDOM_STATE = 42      # 随机种子，保证结果可复现

# 训练/测试集划分比例
TEST_SIZE = 0.3            # 30% 作为测试集
RANDOM_STATE = 42           # 划分的随机种子


def load_data():
    """
    加载预先提取并保存的 HOG 特征和标签
    :return: X (特征矩阵), y (标签向量)
    """
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(LABELS_FILE):
        print('[错误] 未找到特征文件，请先运行 extract_features.py')
        sys.exit(1)

    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)
    print(f'已加载特征: X={X.shape}, y={y.shape}')
    return X, y


def train_svm(X_train, y_train, kernel='rbf', C=10.0, gamma='scale'):
    """
    【核心函数】训练 SVM 分类器
    =============================
    使用 SVC(probability=False) 加速训练。
    predict_proba 不可用 —— demo.py 改用 decision_function + softmax。

    :param X_train: 训练集特征矩阵
    :param y_train: 训练集标签
    :param kernel:  核函数类型
    :param C:       惩罚参数
    :param gamma:   RBF 核参数
    :return: 训练好的 SVC 模型
    """
    svm = SVC(
        kernel=kernel, C=C, gamma=gamma,
        probability=False,           # 关闭概率以大幅加速训练
        random_state=SVM_RANDOM_STATE,
        decision_function_shape='ovo',
        class_weight='balanced',
        cache_size=2000,             # 增大核矩阵缓存（MB），减少重复计算
        max_iter=200000,
    )
    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, scaler, X_test, y_test, class_names):
    """
    【评测函数】在测试集上评估模型性能
    ===================================
    输出：
      - 准确率 (Accuracy)
      - 精确率 (Precision)、召回率 (Recall)、F1-score（各类别 + 整体）
    """
    # 1. 标准化
    X_test_scaled = scaler.transform(X_test)

    # 2. 预测
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f'\n{"="*55}')
    print(f'  测试集准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)')
    print(f'{"="*55}')

    # 3. 详细分类报告
    #    Precision = TP / (TP + FP)  —— 预测为正的样本中有多少是真的正
    #    Recall    = TP / (TP + FN)  —— 所有真正的样本中有多少被正确预测
    #    F1-score  = 2*P*R / (P+R)   —— Precision 和 Recall 的调和平均
    print('\n详细分类报告：')
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4
    ))

    return y_pred, acc


def augment_training_set(X_train, y_train):
    """
    【数据增强】仅对训练集做增强，避免数据泄露到测试集
    ====================================================
    从 processed_data/ 加载与 X_train 对应的图像，
    对每张图像生成增强版本（翻转/旋转/亮度），
    提取 HOG 特征后并入训练集。

    :return: X_train_aug, y_train_aug —— 增强后的训练集
    """
    from skimage.feature import hog

    # 1. 收集所有预处理后的图像路径（顺序与特征提取时一致）
    proc_dir = FEATURES_DIR
    image_paths = []
    image_labels = []
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
                image_paths.append(os.path.join(class_path, fname))
                image_labels.append(lbl)

    # 2. 加载图像并同步标签（跳过加载失败的图像）
    all_images = []
    valid_labels = []
    for p, lbl in zip(image_paths, image_labels):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            all_images.append(img)
            valid_labels.append(lbl)

    if len(all_images) == 0:
        print('  [警告] 未加载到任何图像，跳过数据增强')
        return X_train, y_train

    all_labels = np.array(valid_labels)

    # 3. 用相同的 random_state 划分，找到训练集原始图像
    from sklearn.model_selection import train_test_split as tts
    indices = np.arange(len(all_labels))
    train_idx, _ = tts(
        indices, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=all_labels
    )

    # 4. 只对训练集的图像做增强
    X_aug_list = list(X_train)
    y_aug_list = list(y_train)

    print(f'  正在增强 {len(train_idx)} 张训练图像...')

    for idx in train_idx:
        img = all_images[idx]
        lbl = all_labels[idx]

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 水平翻转
        flipped = cv2.flip(img, 1)
        feat_f = hog(flipped, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(3, 3), visualize=False, channel_axis=None)
        X_aug_list.append(feat_f.astype(np.float32))
        y_aug_list.append(lbl)

        # 小角度旋转 ±10°
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=0)
            feat = hog(rotated, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(3, 3), visualize=False, channel_axis=None)
            X_aug_list.append(feat.astype(np.float32))
            y_aug_list.append(lbl)

        # 亮度变化 ±25
        for delta in [-25, 25]:
            bright = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            feat = hog(bright, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(3, 3), visualize=False, channel_axis=None)
            X_aug_list.append(feat.astype(np.float32))
            y_aug_list.append(lbl)

    X_train_aug = np.array(X_aug_list, dtype=np.float32)
    y_train_aug = np.array(y_aug_list, dtype=np.int32)

    # 检查是否有 NaN/Inf
    if np.any(np.isnan(X_train_aug)) or np.any(np.isinf(X_train_aug)):
        print('  [警告] 增强特征包含 NaN/Inf，已回退到原始训练集')
        return X_train, y_train

    return X_train_aug, y_train_aug


def main():
    # ========== 1. 加载数据 ==========
    print('=' * 55)
    print('  SVM 手势分类器 —— 训练与评估')
    print('=' * 55)

    X, y = load_data()

    # 手势类别名称（用于报告显示）
    class_names = ['fist(握拳)', 'palm(伸掌)', 'thumb(竖拇指)',
                   'ok(OK)', 'victory(比V)']

    # 打印类别分布
    print('\n类别分布：')
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f'  类别 {lbl} - {class_names[lbl]}: {cnt} 张')

    # ========== 2. 划分训练集和测试集 ==========
    # stratify=y 保证训练集和测试集中各类别比例与原始数据一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f'\n原始训练集: {X_train.shape[0]} 样本')
    print(f'原始测试集: {X_test.shape[0]} 样本')

    # ========== 2.5 数据增强（仅训练集）==========
    print('\n★ 对训练集进行数据增强（翻转 + 旋转 ±10° + 亮度 ±25）...')
    X_train, y_train = augment_training_set(X_train, y_train)
    print(f'增强后训练集: {X_train.shape[0]} 样本（约 {X_train.shape[0] / (X_train.shape[0] - X_test.shape[0]) if X_train.shape[0] > X_test.shape[0] else 0:.0f}x）')

    # ========== 3. 特征标准化 ==========
    # SVM 对特征尺度敏感，标准化使每个特征的均值为 0，方差为 1
    # 只对训练集 fit，然后用同样的参数 transform 测试集（防止数据泄露）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # fit_transform = fit(计算均值和方差) + transform(应用标准化)

    # ========== 4. 训练 SVM ==========
    # 安全检查：剔除 NaN/Inf
    assert not np.any(np.isnan(X_train_scaled)), 'X_train 包含 NaN！'
    assert not np.any(np.isinf(X_train_scaled)), 'X_train 包含 Inf！'

    print(f'\n开始训练 SVM (kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})...')
    model = train_svm(X_train_scaled, y_train,
                      kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)

    # 打印训练集上的表现（用于判断是否过拟合）
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f'训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)')

    # ========== 5. 交叉验证 ==========
    # 使用 5 折交叉验证评估模型稳定性
    print('\n执行 5 折交叉验证...')
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f'交叉验证分数: {cv_scores}')
    print(f'平均 CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    # ========== 6. 测试集评估 ==========
    y_pred, test_acc = evaluate_model(model, scaler, X_test, y_test, class_names)

    # ========== 7. 保存模型 ==========
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'\n模型已保存到: {MODEL_FILE}')
    print(f'标准化器已保存到: {SCALER_FILE}')

    # ========== 8. 保存测试预测结果 ==========
    np.save(os.path.join(FEATURES_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(FEATURES_DIR, 'y_pred.npy'), y_pred)
    print(f'测试结果已保存: y_test.npy, y_pred.npy')

    return model, scaler, X_test, y_test, y_pred


if __name__ == '__main__':
    main()
