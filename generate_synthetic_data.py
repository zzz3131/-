# -*- coding: utf-8 -*-
"""
合成手势数据生成器 —— 用于在采集真实数据之前测试整个流水线
==============================================================
使用 OpenCV 绘图函数模拟 5 种手势的简化轮廓，生成 64×64 灰度图。
NOTE: 合成数据仅用于验证代码流程是否正确，实际训练必须使用真实手势图像！

手势设计：
    0_fist:    握拳 —— 圆形 + 横向椭圆（拳头形状）
    1_palm:    伸掌 —— 大椭圆 + 5 根手指线
    2_thumb:   竖拇指 —— 竖直椭圆 + 上方突出短横线
    3_ok:      OK —— 圆圈（拇指和食指形成）+ 三根竖线
    4_victory: 比V —— 椭圆 + 两根斜向上的手指
"""

import cv2
import numpy as np
import os
import random

# ==================== 配置参数 ====================
DATA_DIR = 'data'
IMAGE_SIZE = 64           # 输出图像尺寸
SAMPLES_PER_CLASS = 30    # 每类生成样本数（建议至少 30 用于测试流程）
VARIATION_LEVEL = 0.3     # 随机变化程度（位置、角度、大小的随机扰动）


def add_variation(img, level=0.3):
    """
    添加随机变化模拟真实场景的多样性
    - 随机平移、旋转、缩放
    - 添加随机噪声
    - 随机改变笔画粗细
    """
    h, w = img.shape[:2]

    # 随机旋转（±level * 30 度）
    angle = random.uniform(-level * 30, level * 30)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, (w, h), borderValue=0)

    # 随机平移（±level * 8 像素）
    dx = int(random.uniform(-level * 8, level * 8))
    dy = int(random.uniform(-level * 8, level * 8))
    trans_mat = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, trans_mat, (w, h), borderValue=0)

    # 轻微缩放
    scale = random.uniform(1 - level * 0.2, 1 + level * 0.2)
    scaled_size = (int(w * scale), int(h * scale))
    img = cv2.resize(img, scaled_size)
    if scale > 1:
        # 裁剪中心
        start = (scaled_size[0] - w) // 2
        img = img[start:start+w, start:start+h]
    else:
        # 填充到原大小
        pad = (w - scaled_size[0]) // 2
        padded = np.zeros((h, w), dtype=np.uint8)
        padded[pad:pad+scaled_size[1], pad:pad+scaled_size[0]] = img
        img = padded

    # 添加高斯噪声
    noise = np.random.normal(0, 10 * level, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 随机笔画粗细（膨胀/腐蚀）
    thickness = random.choice([1, 2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    if random.random() > 0.5:
        img = cv2.dilate(img, kernel, iterations=1)
    else:
        img = cv2.erode(img, kernel, iterations=1)

    return img


def draw_fist():
    """画握拳 —— 接近圆形的椭圆 + 上方一些小弧线表示指关节"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    # 主体（类圆形椭圆）
    cv2.ellipse(img, (cx, cy), (18, 22), 0, 0, 360, 255, -1)
    # 指关节线条
    for angle in [-20, 0, 20]:
        rad = np.radians(angle)
        x = int(cx + 16 * np.sin(rad))
        y = int(cy - 16 * np.cos(rad))
        cv2.circle(img, (x, y), 4, 200, -1)
    return img


def draw_palm():
    """画伸掌 —— 椭圆手掌 + 5 根手指线"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2 + 4
    # 手掌（椭圆）
    cv2.ellipse(img, (cx, cy), (16, 18), 0, 0, 360, 255, -1)
    # 5 根手指（从手掌上方伸出）
    for dx in [-10, -5, 0, 5, 10]:
        x = cx + dx
        cv2.line(img, (x, cy - 14), (x + dx//4, cy - 30), 255, 4)
        cv2.circle(img, (x + dx//4, cy - 30), 3, 255, -1)
    return img


def draw_thumb():
    """画竖拇指 —— 竖直椭圆拳头 + 一根向上的拇指"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2 + 4
    # 拳头主体
    cv2.ellipse(img, (cx, cy), (15, 18), 0, 0, 360, 255, -1)
    # 竖起的拇指（从右侧伸出向上）
    cv2.line(img, (cx + 10, cy - 8), (cx + 12, cy - 28), 255, 6)
    cv2.circle(img, (cx + 12, cy - 28), 5, 255, -1)
    # 其他手指蜷缩的痕迹
    for dy in [0, 4, 8]:
        cv2.line(img, (cx - 4, cy - 8 + dy), (cx + 6, cy - 10 + dy), 200, 2)
    return img


def draw_ok():
    """画 OK —— 圆形（拇指和食指形成）+ 三根竖线（其余手指）"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    # 手掌
    cv2.ellipse(img, (cx, cy), (14, 16), 0, 0, 360, 200, -1)
    # 拇指和食指形成的圆圈
    cv2.circle(img, (cx + 8, cy - 16), 8, 255, 3)
    # 三根伸展的手指
    for dx in [-4, 0, 4]:
        cv2.line(img, (cx + dx, cy + 6), (cx + dx, cy + 28), 255, 4)
        cv2.circle(img, (cx + dx, cy + 28), 2, 255, -1)
    # 拇指根部
    cv2.line(img, (cx + 14, cy - 4), (cx + 16, cy - 8), 255, 5)
    return img


def draw_victory():
    """画比V —— 椭圆拳头 + 两根斜向上的手指（V 字形）"""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2 + 4
    # 拳头主体
    cv2.ellipse(img, (cx, cy), (15, 16), 0, 0, 360, 255, -1)
    # 两根手指呈 V 形
    cv2.line(img, (cx - 8, cy - 10), (cx - 16, cy - 28), 255, 5)
    cv2.circle(img, (cx - 16, cy - 28), 4, 255, -1)
    cv2.line(img, (cx + 8, cy - 10), (cx + 16, cy - 28), 255, 5)
    cv2.circle(img, (cx + 16, cy - 28), 4, 255, -1)
    # 蜷缩的手指
    for dx in [-3, 0, 3]:
        cv2.circle(img, (cx + dx, cy + 4), 4, 200, -1)
    return img


def generate_all():
    """生成所有类别的合成数据"""
    print('=' * 55)
    print('  合成手势数据生成')
    print('=' * 55)
    print(f'  每类生成: {SAMPLES_PER_CLASS} 张')
    print(f'  图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}')
    print(f'  随机变化程度: {VARIATION_LEVEL}')
    print()

    draw_functions = {
        '0_fist':     draw_fist,
        '1_palm':     draw_palm,
        '2_thumb':    draw_thumb,
        '3_ok':       draw_ok,
        '4_victory':  draw_victory,
    }

    for class_name, draw_fn in draw_functions.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            # 生成基础手势图像
            img = draw_fn()
            # 添加随机变化（模拟不同拍摄角度、位置）
            img = add_variation(img, VARIATION_LEVEL)

            # 保存
            fname = f'synthetic_{i:04d}.jpg'
            cv2.imwrite(os.path.join(class_dir, fname), img)

        print(f'  [{class_name}] 已生成 {SAMPLES_PER_CLASS} 张合成图像')

    print(f'\n  完成！合成数据保存在 {DATA_DIR}/ 下各子文件夹')


if __name__ == '__main__':
    generate_all()
