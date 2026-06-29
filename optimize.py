# -*- coding: utf-8 -*-
"""
模型优化模块 —— LBP + HOG 特征融合 & SVM 超参数网格搜索
============================================================
核心概念：
    1. LBP（Local Binary Pattern，局部二值模式）：
       - 描述图像局部纹理特征
       - 对每个像素，将其周围 8 个邻域像素与该像素比较：
         > 该像素 → 标记为 1，否则标记为 0 → 得到 8 位二进制数 = LBP 编码
       - 统计整幅图像中各 LBP 编码出现的频率 → LBP 直方图特征
       - LBP 对光照变化非常鲁棒（因为是比较相对大小，而非绝对值）

    2. 特征融合（Feature Fusion）：
       - HOG: 描述形状/边缘方向 → 捕捉手势的轮廓和整体形状
       - LBP: 描述纹理/局部模式 → 捕捉皮肤纹理、手指褶皱等细节
       - 串联融合：直接拼接两个特征向量 → 互补信息
         HOG(2916维) + LBP(n_bins维) = 融合特征

    3. 网格搜索（Grid Search）：
       系统地遍历所有超参数组合，通过交叉验证找到最优参数。
       - C:     [0.1, 1, 10, 100]       —— 惩罚系数
       - gamma: ['scale', 'auto', 0.01, 0.001] —— RBF 核参数
       共 4 × 4 = 16 种组合，每种做 5 折交叉验证 → 80 次训练

    4. 为什么要做这些优化？
       - HOG 可能对某些手势（如 OK 和握拳）区分度不够
       - LBP 补充了纹理信息，可能提高区分度
       - 默认超参数不一定最优，网格搜索能找到更好的参数组合
"""

import numpy as np
import os
import pickle
import sys
import time
from skimage.feature import local_binary_pattern

from sklearn.svm import SVC
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==================== 配置参数 ====================
FEATURES_DIR = 'processed_data'
MODEL_DIR = 'models'
RESULT_DIR = 'results'

# LBP 参数
LBP_RADIUS = 2          # LBP 半径（邻域距离）
LBP_N_POINTS = 16       # 邻域采样点数
LBP_METHOD = 'uniform'  # LBP 模式：'uniform' 使用均匀模式（旋转不变 + 降维）

# 网格搜索参数
GRID_CV_FOLDS = 5       # 交叉验证折数

TEST_SIZE = 0.3
RANDOM_STATE = 42


# ==================== LBP 特征提取 ====================

def extract_lbp_features_batch(images, radius=2, n_points=16, method='uniform'):
    """
    【核心函数1】对一批图像提取 LBP 特征
    ======================================
    LBP 计算流程：
      1. 对每个像素，比较其与周围 n_points 个邻域像素的灰度值
      2. 邻域 > 中心 → 1，否则 → 0
      3. 按顺序排列 → 一个 n_points 位的二进制数 = LBP 码
      4. 统计整幅图中各 LBP 码的频率 → 直方图
      5. 对直方图做 L2 归一化

    :param images:  图像列表，每张为 64×64 灰度图
    :param radius:  LBP 邻域半径
    :param n_points: 邻域采样点数量
    :param method:  'uniform' 使用均匀 LBP 模式（旋转不变的子集）
    :return: lbp_features —— (n_samples, n_bins) 的特征矩阵
    """
    features_list = []

    # LBP 均匀模式下的 bin 数量：
    # 均匀模式将 2^n_points 种可能的 LBP 码压缩为 n_points*(n_points-1)+3 种
    n_bins = n_points * (n_points - 1) + 3  # 例如 n_points=16 → 243 bins

    for img in images:
        # 步骤1: 计算每个像素的 LBP 码
        lbp = local_binary_pattern(img, n_points, radius, method=method)

        # 步骤2: 统计 LBP 直方图
        # 使用 n_bins 个 bin，范围 [0, n_bins]
        hist, _ = np.histogram(lbp.ravel(),
                               bins=n_bins,
                               range=(0, n_bins),
                               density=True)  # density=True 做 L1 归一化
        features_list.append(hist)

    return np.array(features_list, dtype=np.float32)


# ==================== Helper ====================

def _create_svc(C=10.0, gamma='scale', kernel='rbf', probability=True):
    """
    Create an SVC with probability support.
    Note: probability=True is deprecated in sklearn >= 1.9 but still works.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning,
                                message='.*probability.*')
        return SVC(kernel=kernel, C=C, gamma=gamma,
                   probability=probability, random_state=RANDOM_STATE,
                   class_weight='balanced')


# ==================== 网格搜索 ====================

def grid_search_svm(X_train, y_train):
    """
    【核心函数2】SVM 超参数网格搜索
    ==================================
    通过 GridSearchCV 遍历所有参数组合，使用交叉验证找到最优参数。

    搜索的参数网格：
      - C:      控制间隔与训练误差的权衡
      - gamma:  RBF 核的宽度参数
      - kernel: 核函数类型（可选，此处专注于 RBF）

    :return: best_params (最优参数), cv_results (详细交叉验证结果)
    """
    print('\n' + '=' * 55)
    print('  网格搜索 —— SVM 超参数优化')
    print('=' * 55)

    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
        'kernel': ['rbf'],
    }

    total_combinations = (len(param_grid['C']) *
                          len(param_grid['gamma']) *
                          len(param_grid['kernel']))
    print(f'  参数组合数: {total_combinations}')
    print(f'  交叉验证: {GRID_CV_FOLDS} 折')
    print(f'  总训练次数: {total_combinations * GRID_CV_FOLDS}')

    # 创建 GridSearchCV
    # n_jobs=-1 使用所有 CPU 核心并行训练
    # verbose=2 显示详细训练进度
    grid = GridSearchCV(
        _create_svc(),
        param_grid,
        cv=StratifiedKFold(n_suffix=GRID_CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE),
        scoring='accuracy',      # 以准确率为优化目标
        n_jobs=-1,               # 并行加速
        verbose=1,                # 显示进度
        return_train_score=True,
    )

    print('\n开始搜索... (可能需要几分钟)')
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f'搜索完成，耗时: {elapsed:.1f} 秒')

    print(f'\n最优参数: {grid.best_params_}')
    print(f'最优交叉验证准确率: {grid.best_score_:.4f}')

    return grid


# ==================== 比较实验 ====================

def run_comparison_experiment():
    """
    【主实验】对比不同特征和参数组合的性能
    =========================================
    实验组：
      1. HOG 特征 + 默认 SVM 参数 (baseline)
      2. HOG 特征 + 网格搜索最优参数
      3. LBP 特征 + 默认 SVM 参数
      4. HOG + LBP 融合特征 + 默认 SVM 参数
      5. HOG + LBP 融合特征 + 网格搜索最优参数 (最优方案)

    输出：对比表格，便于写入报告
    """
    print('=' * 55)
    print('  模型优化实验 —— 特征融合 + 超参数优化')
    print('=' * 55)

    # ======== 1. 加载数据 ========
    print('\n[1/5] 加载原始图像和标签...')
    X = np.load(os.path.join(FEATURES_DIR, 'features_hog.npy'))
    y = np.load(os.path.join(FEATURES_DIR, 'labels.npy'))
    print(f'  HOG 特征: {X.shape}')

    # 加载预处理后的图像（用于提取 LBP）
    print('\n[2/5] 加载预处理图像并提取 LBP 特征...')
    images = []
    labels_for_images = []
    for class_name in sorted(os.listdir(FEATURES_DIR)):
        class_dir = os.path.join(FEATURES_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        try:
            label = int(class_name.split('_')[0])
        except ValueError:
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, fname)
                import cv2
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels_for_images.append(label)

    # 提取 LBP 特征
    X_lbp = extract_lbp_features_batch(images, LBP_RADIUS, LBP_N_POINTS, LBP_METHOD)
    print(f'  LBP 特征: {X_lbp.shape}')

    # 确保 LBP 和 HOG 的顺序一致（都按目录遍历，顺序应一致）
    assert X.shape[0] == X_lbp.shape[0], \
        f'HOG({X.shape[0]}) 与 LBP({X_lbp.shape[0]}) 样本数不一致！'

    # HOG + LBP 融合：直接拼接
    X_fused = np.hstack([X, X_lbp])
    print(f'  HOG+LBP 融合特征: {X_fused.shape}')

    # ======== 2. 划分训练/测试集 ========
    X_hog_train, X_hog_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    _, X_lbp_test, _, _ = train_test_split(
        X_lbp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    _, X_fused_test, _, _ = train_test_split(
        X_fused, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # LBP 的训练集
    X_lbp_train = X_lbp[:len(X_hog_train)]
    X_fused_train = X_fused[:len(X_hog_train)]

    # ======== 3. 标准化 ========
    scaler_hog = StandardScaler()
    X_hog_train_s = scaler_hog.fit_transform(X_hog_train)
    X_hog_test_s = scaler_hog.transform(X_hog_test)

    scaler_lbp = StandardScaler()
    X_lbp_train_s = scaler_lbp.fit_transform(X_lbp_train)
    X_lbp_test_s = scaler_lbp.transform(X_lbp_test)

    scaler_fused = StandardScaler()
    X_fused_train_s = scaler_fused.fit_transform(X_fused_train)
    X_fused_test_s = scaler_fused.transform(X_fused_test)

    # ======== 4. 运行实验 ========
    class_names = ['fist', 'palm', 'thumb', 'ok', 'victory']
    results = []

    print('\n[3/5] 实验 1: HOG + SVM 默认参数 (Baseline)...')
    svm1 = _create_svc(kernel='rbf', C=10.0, gamma='scale')
    svm1.fit(X_hog_train_s, y_train)
    acc1 = accuracy_score(y_test, svm1.predict(X_hog_test_s))
    results.append(('HOG', '默认 (C=10)', acc1))

    print('\n[4/5] 实验 2: HOG + SVM 网格搜索...')
    grid = grid_search_svm(X_hog_train_s, y_train)
    best = grid.best_estimator_
    acc2 = accuracy_score(y_test, best.predict(X_hog_test_s))
    results.append(('HOG', f'网格搜索 C={grid.best_params_["C"]}, γ={grid.best_params_["gamma"]}', acc2))

    print('\n[5/5] 实验 3: LBP + SVM 默认参数...')
    svm3 = _create_svc(kernel='rbf', C=10.0, gamma='scale')
    svm3.fit(X_lbp_train_s, y_train)
    acc3 = accuracy_score(y_test, svm3.predict(X_lbp_test_s))
    results.append(('LBP', '默认 (C=10)', acc3))

    print('\n实验 4: HOG+LBP + SVM 默认参数...')
    svm4 = _create_svc(kernel='rbf', C=10.0, gamma='scale')
    svm4.fit(X_fused_train_s, y_train)
    acc4 = accuracy_score(y_test, svm4.predict(X_fused_test_s))
    results.append(('HOG+LBP', '默认 (C=10)', acc4))

    print('\n实验 5: HOG+LBP + SVM 网格搜索...')
    # 对融合特征做网格搜索
    grid_fused = GridSearchCV(
        _create_svc(),
        {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.001],
         'kernel': ['rbf']},
        cv=StratifiedKFold(n_splits=GRID_CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE),
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_fused.fit(X_fused_train_s, y_train)
    best_fused = grid_fused.best_estimator_
    acc5 = accuracy_score(y_test, best_fused.predict(X_fused_test_s))
    results.append(('HOG+LBP', f'网格搜索 C={grid_fused.best_params_["C"]}, γ={grid_fused.best_params_["gamma"]}', acc5))

    # ======== 5. 输出对比结果 ========
    print('\n' + '=' * 70)
    print('  实验结果对比')
    print('=' * 70)
    print(f'  {"特征":<12s} {"参数":<35s} {"测试准确率":>12s}')
    print('-' * 70)
    for feature, params, acc in results:
        print(f'  {feature:<12s} {params:<35s} {acc*100:>10.2f}%')
    print('-' * 70)

    # 找出最佳方案
    best_result = max(results, key=lambda x: x[2])
    print(f'\n  ★ 最佳方案: {best_result[0]} + {best_result[1]}')
    print(f'    准确率: {best_result[2]*100:.2f}%')

    # ======== 6. 保存结果 ========
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.savez(os.path.join(RESULT_DIR, 'comparison_results.npz'),
             results=np.array(results, dtype=object))

    # 保存最优模型
    # 找到 HOG+LBP 网格搜索的最佳模型
    best_model = best_fused if acc5 >= max(acc1, acc2, acc3, acc4) else grid.best_estimator_
    with open(os.path.join(MODEL_DIR, 'svm_optimized.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    # 保存融合特征所需的标准化器
    best_scaler = scaler_fused if acc5 >= max(acc1, acc2, acc3, acc4) else scaler_hog
    with open(os.path.join(MODEL_DIR, 'scaler_optimized.pkl'), 'wb') as f:
        pickle.dump(best_scaler, f)

    print(f'\n最优模型已保存: {MODEL_DIR}/svm_optimized.pkl')
    print(f'对比结果已保存: {RESULT_DIR}/comparison_results.npz')

    return results


def main():
    run_comparison_experiment()


if __name__ == '__main__':
    main()
