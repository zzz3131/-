# -*- coding: utf-8 -*-
"""
实时手势识别演示脚本
=======================
功能：打开摄像头，实时检测手部区域，提取特征，用训练好的 SVM 模型分类。

使用前提：
    1. 已运行 extract_features.py 提取特征
    2. 已运行 train_svm.py 训练好模型
    3. 连接摄像头

操作：
    - 将手放在画面中，模型实时识别手势
    - 按 'q' 退出
    - 按 's' 截屏保存当前帧
"""

import cv2
import numpy as np
import os
import pickle
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入自定义预处理模块
from preprocess import (ycbcr_skin_mask, morphological_refine,
                        extract_hand_region)

# ==================== 配置参数 ====================
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'svm_gesture.pkl')
SCALER_FILE = os.path.join(MODEL_DIR, 'scaler.pkl')

# 手势名称（与数据采集时一致）
CLASS_NAMES = [
    'fist (握拳)',
    'palm (伸掌)',
    'thumb (竖拇指)',
    'OK',
    'victory (比V)',
]

# 画面中手势区域的尺寸
ROI_SIZE = 200

# 颜色映射（不同手势用不同颜色显示）
COLORS = [
    (0, 0, 255),     # 握拳 - 红色
    (0, 255, 0),     # 伸掌 - 绿色
    (255, 0, 0),     # 竖拇指 - 蓝色
    (255, 255, 0),   # OK - 青色
    (255, 0, 255),   # 比V - 紫色
]


def load_model():
    """加载训练好的 SVM 模型和标准化器"""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print('[错误] 未找到模型文件，请先运行 train_svm.py')
        sys.exit(1)

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    print(f'模型已加载: {MODEL_FILE}')
    return model, scaler


def extract_hog_features(image):
    """
    对单张 64×64 灰度图提取 HOG 特征
    使用与 extract_features.py 相同的参数
    """
    from skimage.feature import hog
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(3, 3),
        visualize=False,
        channel_axis=None
    )
    return features.reshape(1, -1).astype(np.float32)


def main():
    print('=' * 55)
    print('  实时手势识别演示')
    print('=' * 55)

    # 1. 加载模型
    model, scaler = load_model()

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print('\n操作说明：')
    print('  将手放入绿色方框内，自动识别手势')
    print("  按 'q' 退出，按 's' 截图")
    print('-' * 55)

    # 用于平滑预测结果的滑动窗口
    prediction_history = []
    HISTORY_SIZE = 5  # 最近5帧的预测结果

    while True:
        # 3. 读取帧
        ret, frame = cap.read()
        if not ret:
            continue

        # 4. 镜像翻转
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 5. 绘制 ROI 框
        x1 = w // 2 - ROI_SIZE // 2
        y1 = h // 2 - ROI_SIZE // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 6. 提取 ROI 区域（手部应在其中）
        roi = frame[y1:y2, x1:x2].copy()

        # 7. 肤色检测 + 手部提取
        skin_mask = ycbcr_skin_mask(roi)
        refined_mask = morphological_refine(skin_mask)
        hand_region = extract_hand_region(roi, refined_mask)

        # 8. 转灰度 + 直方图均衡 + 缩放
        hand_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        # ★ CLAHE 直方图均衡（与 preprocess.py 保持一致）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hand_gray = clahe.apply(hand_gray)
        hand_resized = cv2.resize(hand_gray, (64, 64),
                                  interpolation=cv2.INTER_AREA)

        # 9. 提取 HOG 特征
        hog_feat = extract_hog_features(hand_resized)

        # 10. 标准化 + 预测
        hog_feat_scaled = scaler.transform(hog_feat)
        pred_label = model.predict(hog_feat_scaled)[0]
        # 用 decision_function + softmax 替代 predict_proba（训练时未开概率估计以加速）
        decision_scores = model.decision_function(hog_feat_scaled)[0]
        # softmax: exp(s) / sum(exp(s))
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        pred_proba = exp_scores / exp_scores.sum()

        # 11. 滑动窗口平滑（减少闪烁）
        prediction_history.append(pred_label)
        if len(prediction_history) > HISTORY_SIZE:
            prediction_history.pop(0)
        # 取最近几帧中出现最多的预测
        smoothed_label = max(set(prediction_history),
                             key=prediction_history.count)

        # 12. 获取置信度
        confidence = pred_proba[smoothed_label]

        # 13. 在显示画面中绘制结果
        color = COLORS[smoothed_label]
        name = CLASS_NAMES[smoothed_label]

        # 在 ROI 上方显示识别结果
        cv2.putText(display, f'{name}',
                    (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3)
        cv2.putText(display, f'confidence: {confidence:.2f}',
                    (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

        # 在左下角显示各类别概率
        proba_text_y = h - 20
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, pred_proba)):
            bar_len = int(prob * 200)
            cv2.putText(display, f'{name}:',
                        (10, proba_text_y - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 1)
            cv2.rectangle(display,
                          (120, proba_text_y - i * 25 - 8),
                          (120 + bar_len, proba_text_y - i * 25 + 2),
                          COLORS[i], -1)
            cv2.putText(display, f'{prob:.2f}',
                        (125 + bar_len, proba_text_y - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 显示肤色掩码的小窗口（右上角）
        skin_display = cv2.resize(
            cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR),
            (100, 100))
        display[10:110, w-110:w-10] = skin_display
        cv2.putText(display, 'Skin', (w-100, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示预处理后的手部图像
        hand_display = cv2.resize(
            cv2.cvtColor(hand_resized, cv2.COLOR_GRAY2BGR),
            (64, 64))
        display[10:74, w-180:w-116] = hand_display
        cv2.putText(display, 'HOG input', (w-175, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 14. 显示
        cv2.imshow('Gesture Recognition Demo', display)

        # 15. 键盘处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 截图保存
            timestamp = cv2.getTickCount()
            cv2.imwrite(f'screenshot_{timestamp}.jpg', display)
            print(f'截图已保存')

    # 16. 清理
    cap.release()
    cv2.destroyAllWindows()
    print('演示结束。')


if __name__ == '__main__':
    main()
