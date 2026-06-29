# -*- coding: utf-8 -*-
"""
预处理模块 —— YCbCr 肤色检测 + 手部区域提取 + 统一缩放
===========================================================
核心概念：
    1. YCbCr 色彩空间：
       - Y  = 亮度（Luma），受光照影响大
       - Cb = 蓝色色度分量
       - Cr = 红色色度分量
       - 肤色在 Cb-Cr 平面上聚类非常集中，因此对光照变化有一定鲁棒性

    2. 肤色分割流程：
       RGB → YCbCr → 阈值分割(Cb, Cr范围) → 形态学处理 → 轮廓提取
       → 最大连通域 = 手部区域 → 裁剪缩放为 64×64 灰度图

参考文献：
    - 肤色在 YCbCr 空间的经验阈值：
      77 < Cb < 127, 133 < Cr < 173  (多种文献的常见范围)
"""

import cv2
import numpy as np
import os
import sys

# ==================== 配置参数 ====================
# 输入/输出目录
RAW_DATA_DIR = 'data'                 # 原始采集图像路径
PROCESSED_DATA_DIR = 'processed_data'  # 预处理后图像保存路径

# 最终输出图像的尺寸（HOG 特征提取要求统一尺寸）
TARGET_SIZE = (64, 64)

# YCbCr 肤色阈值范围
# Cb: 蓝色色度分量 —— 肤色通常在这个范围内
CB_LOWER = 77
CB_UPPER = 127
# Cr: 红色色度分量 —— 肤色偏红，Cr 值较高
CR_LOWER = 133
CR_UPPER = 173

# 形态学处理参数
MORPH_KERNEL_SIZE = 5      # 形态学操作核大小
MORPH_CLOSE_ITER = 2       # 闭运算迭代次数（先膨胀再腐蚀，填充空洞）
MORPH_OPEN_ITER = 1        # 开运算迭代次数（先腐蚀再膨胀，去除噪点）

# 轮廓过滤参数
MIN_CONTOUR_AREA = 500     # 最小轮廓面积（像素），小于此值的区域视为噪声


def ycbcr_skin_mask(rgb_image):
    """
    【核心函数1】YCbCr 肤色检测 —— 生成肤色二值掩码
    =====================================================
    原理：
        1. 将 RGB 图像转换到 YCbCr 色彩空间
        2. 对 Cb 和 Cr 两个通道分别设定阈值
        3. 两张二值图取交集，得到肤色区域

    为什么用 YCbCr 而不是 RGB 或 HSV？
        - RGB 三个通道高度相关，光照变化导致三个通道同时变化，难以设定阈值
        - HSV 中 H（色调）对光照较鲁棒，但肤色在不同光照下 H 值范围较窄且不稳定
        - YCbCr 将亮度（Y）与色度（Cb, Cr）分离，肤色在 Cb-Cr 平面上非常集中，
          对光照变化（只影响 Y 通道）有天然的鲁棒性

    :param rgb_image: BGR 格式的输入图像（OpenCV 默认读取格式）
    :return: skin_mask —— 二值掩码（肤色区域=255，非肤色=0）
    """
    # 步骤1: BGR → YCbCr 色彩空间转换
    # OpenCV 的 COLOR_BGR2YCrCb 输出顺序为 Y, Cr, Cb（注意 Cr 和 Cb 的顺序！）
    ycrcb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)

    # 步骤2: 分离三个通道
    # 通道0 = Y（亮度），通道1 = Cr（红色色度），通道2 = Cb（蓝色色度）
    y, cr, cb = cv2.split(ycrcb)

    # 步骤3: 对 Cb 通道做阈值分割
    # Cb 值在 77~127 之间的是肤色（经验值）
    _, cb_mask = cv2.threshold(cb, CB_UPPER, 255, cv2.THRESH_BINARY_INV)
    _, cb_mask_low = cv2.threshold(cb, CB_LOWER, 255, cv2.THRESH_BINARY)
    cb_mask = cv2.bitwise_and(cb_mask, cb_mask_low)
    # 等价于: cb_mask = (cb > CB_LOWER) & (cb < CB_UPPER)

    # 步骤4: 对 Cr 通道做阈值分割
    # Cr 值在 133~173 之间的是肤色（经验值）
    _, cr_mask = cv2.threshold(cr, CR_UPPER, 255, cv2.THRESH_BINARY_INV)
    _, cr_mask_low = cv2.threshold(cr, CR_LOWER, 255, cv2.THRESH_BINARY)
    cr_mask = cv2.bitwise_and(cr_mask, cr_mask_low)
    # 等价于: cr_mask = (cr > CR_LOWER) & (cr < CR_UPPER)

    # 步骤5: 两个通道的掩码取交集 = 最终肤色区域
    skin_mask = cv2.bitwise_and(cb_mask, cr_mask)

    return skin_mask


def morphological_refine(binary_mask):
    """
    【核心函数2】形态学后处理 —— 去除噪声、填充空洞
    ====================================================
    操作顺序：闭运算（填充空洞）→ 开运算（去除孤立噪点）

    形态学基础：
        - 膨胀(dilate):   核覆盖区域取最大值 → 白色区域扩大
        - 腐蚀(erode):    核覆盖区域取最小值 → 白色区域缩小
        - 闭运算(close):  先膨胀后腐蚀 → 填充前景中的小黑洞
        - 开运算(open):   先腐蚀后膨胀 → 去除前景中的小白点

    :param binary_mask: 二值掩码
    :return: refined_mask —— 处理后的二值掩码
    """
    # 创建椭圆形结构元素（核），比方形核更平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    # 闭运算：先膨胀(扩大白色区域)再腐蚀(缩小白色区域)
    # 作用：填充肤色区域内的空洞（如指甲、阴影造成的小黑洞）
    refined = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel,
                               iterations=MORPH_CLOSE_ITER)

    # 开运算：先腐蚀再膨胀
    # 作用：去除背景中的孤立噪点（如墙上的肤色点状误检）
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel,
                               iterations=MORPH_OPEN_ITER)

    return refined


def extract_hand_region(rgb_image, skin_mask):
    """
    【核心函数3】提取手部区域 —— 找最大连通域并裁剪
    ====================================================
    原理：
        1. 在肤色掩码上找所有轮廓（连通域）
        2. 选择面积最大的轮廓（因为手通常是画面中最大的肤色区域）
        3. 用该轮廓的边界框裁剪原图
        4. 如果找不到任何轮廓（肤色检测完全失败），返回原图

    :param rgb_image: BGR 格式原图
    :param skin_mask: 肤色二值掩码
    :return: hand_crop —— 裁剪后的手部区域（BGR 格式）
    """
    # 步骤1: 查找所有轮廓
    # cv2.RETR_EXTERNAL: 只返回最外层轮廓（不包含洞的轮廓）
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线段，只保留端点
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # 没有找到任何肤色区域 → 返回原图（后续会用整图做特征提取）
        return rgb_image

    # 步骤2: 按轮廓面积降序排序，取最大轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    # 步骤3: 过滤面积过小的轮廓（可能是噪声）
    area = cv2.contourArea(largest_contour)
    if area < MIN_CONTOUR_AREA:
        return rgb_image  # 面积太小，不可靠，返回原图

    # 步骤4: 获取最大轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 步骤5: 添加少量边距（padding），确保手部边缘不被裁掉
    padding = 10
    img_h, img_w = rgb_image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    # 步骤6: 裁剪手部区域
    hand_crop = rgb_image[y1:y2, x1:x2]

    return hand_crop


def preprocess_single(image_path):
    """
    【主流程】对单张图像进行完整的预处理流水线
    ===========================================
    流程：读取 → 肤色检测 → 形态学处理 → 手部提取 → 转灰度 → 直方图均衡 → 缩放

    :param image_path: 原始图像路径
    :return: processed_gray —— 64×64 灰度图，如果处理失败返回 None
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f'  [警告] 无法读取图像: {image_path}')
        return None

    # 2. YCbCr 肤色检测 → 得到肤色掩码
    skin_mask = ycbcr_skin_mask(img)

    # 3. 形态学后处理 → 去除噪声、填充空洞
    refined_mask = morphological_refine(skin_mask)

    # 4. 提取手部区域 → 裁剪
    hand = extract_hand_region(img, refined_mask)

    # ★ 改进：检查肤色检测是否有效（肤色面积占比）
    total_pixels = refined_mask.shape[0] * refined_mask.shape[1]
    skin_pixels = np.count_nonzero(refined_mask)
    skin_ratio = skin_pixels / total_pixels

    # 如果肤色区域太小（<5%）或太大（>90%），说明肤色检测可能失效
    # 此时直接用整张 ROI 图（背景干扰总比完全错误的分割好）
    if skin_ratio < 0.05 or skin_ratio > 0.90:
        # 肤色检测可能失败，但仍对全图做直方图均衡
        # 这比使用错误的分割结果更可靠
        pass  # 继续使用 hand（可能是原图），后续用直方图均衡补偿

    # 5. 转为灰度图
    # 使用加权法: Gray = 0.299*R + 0.587*G + 0.114*B（OpenCV 默认）
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)

    # ★ 改进：直方图均衡化 —— 增强对比度，减少光照变化的影响
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # 比全局直方图均衡更鲁棒，不会过度放大噪声区域
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 6. 缩放为统一尺寸 64×64
    # INTER_AREA 适合缩小图像，质量较好
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return resized


def preprocess_all():
    """
    【批量处理】遍历所有手势类别文件夹，预处理全部图像
    =====================================================
    输出:
        - 处理后的 64×64 灰度图保存到 processed_data/<类别>/
        - 打印处理统计信息
    """
    print('=' * 55)
    print('  图像预处理 —— YCbCr 肤色检测 + 手部区域提取')
    print('=' * 55)

    total_processed = 0
    total_failed = 0

    # 遍历每个手势类别
    for class_name in sorted(os.listdir(RAW_DATA_DIR)):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue  # 跳过非文件夹

        # 创建对应的输出目录
        out_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        os.makedirs(out_dir, exist_ok=True)

        # 获取该类别下所有图像文件
        image_files = [f for f in os.listdir(class_dir)
                       if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        if len(image_files) == 0:
            print(f'\n[{class_name}] 无图像，跳过')
            continue

        print(f'\n[{class_name}] 共 {len(image_files)} 张图像，开始处理...')
        class_ok, class_fail = 0, 0

        for fname in image_files:
            in_path = os.path.join(class_dir, fname)
            # 执行完整预处理流水线
            processed = preprocess_single(in_path)

            if processed is not None:
                # 保存处理后的图像
                out_path = os.path.join(out_dir, fname)
                cv2.imwrite(out_path, processed)
                class_ok += 1
            else:
                class_fail += 1

        print(f'  成功: {class_ok}, 失败: {class_fail}')
        total_processed += class_ok
        total_failed += class_fail

    print(f'\n{"="*55}')
    print(f'  总计: 成功 {total_processed} 张, 失败 {total_failed} 张')
    print(f'  输出目录: {PROCESSED_DATA_DIR}/')
    print(f'{"="*55}')


def demo_skin_detection(image_path):
    """
    【演示函数】对单张图像可视化肤色检测的中间过程
    用于调试和报告中的示意图
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f'无法读取: {image_path}')
        return

    # 肤色掩码
    skin_mask = ycbcr_skin_mask(img)
    # 形态学处理后
    refined = morphological_refine(skin_mask)
    # 肤色区域叠加在原图上
    skin_overlay = cv2.bitwise_and(img, img, mask=refined)

    # 拼接展示
    top = np.hstack([img,
                     cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)])
    bottom = np.hstack([cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR),
                        skin_overlay])
    result = np.vstack([top, bottom])

    cv2.imshow('Skin Detection Demo (Original | Skin Mask | Refined | Overlay)', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 如果提供了图像路径，运行演示模式
        demo_skin_detection(sys.argv[1])
    else:
        # 否则批量处理全部数据
        preprocess_all()
