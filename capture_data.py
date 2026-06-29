# -*- coding: utf-8 -*-
"""
数据采集模块 —— 通过摄像头采集自定义手势图像
====================================================
使用方法：
    python capture_data.py

操作说明：
    - 按对应数字键 (0~4) 选择当前要采集的手势类别
    - 按 空格键  拍摄一张图片并自动保存到对应文件夹
    - 按 'q' 键  退出程序

手势类别（共5种）：
    0 = 握拳 (fist)
    1 = 伸掌 (palm)
    2 = 竖拇指 (thumbs up)
    3 = OK (ok)
    4 = 比V (victory)

每类手势建议采集 50~100 张，在不同光照、角度、背景下拍摄以提高泛化能力。
"""

import cv2
import os
import time

# ==================== 配置参数 ====================
# 数据存储根目录
DATA_DIR = 'data'
# 手势类别名称（与文件夹名对应）
CLASSES = {
    0: '0_fist',      # 握拳
    1: '1_palm',      # 伸掌
    2: '2_thumb',     # 竖拇指
    3: '3_ok',        # OK
    4: '4_victory',   # 比V
}
# 拍摄区域（ROI）大小 —— 预处理前的统一尺寸
ROI_SIZE = 200
# 两次拍摄之间的冷却时间（秒），防止重复拍摄
COOLDOWN = 0.3


def create_directories():
    """
    创建所有数据存储目录
    如果目录已存在则跳过，否则新建
    """
    for folder in CLASSES.values():
        folder_path = os.path.join(DATA_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)
        # 统计已有图片数量
        count = len([f for f in os.listdir(folder_path)
                     if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f'  {folder}: 已有 {count} 张图片')


def get_next_filename(class_id):
    """
    为新图片生成不重复的文件名
    命名规则：gesture_<类别id>_<序号>.jpg
    """
    folder = os.path.join(DATA_DIR, CLASSES[class_id])
    # 找到当前最大序号
    existing = [f for f in os.listdir(folder)
                if f.endswith(('.jpg', '.png', '.jpeg'))]
    max_idx = -1
    for f in existing:
        try:
            # 从文件名中提取序号
            idx = int(f.split('_')[-1].split('.')[0])
            max_idx = max(max_idx, idx)
        except ValueError:
            continue
    return os.path.join(folder, f'gesture_{class_id}_{max_idx + 1:04d}.jpg')


def main():
    print('=' * 55)
    print('  静态手势图像采集程序')
    print('=' * 55)
    print('操作说明：')
    for k, v in CLASSES.items():
        print(f'  按 [{k}] → 切换到: {v}')
    print('  按 [空格] → 拍摄当前手势')
    print('  按 [q]   → 退出')
    print('-' * 55)

    # 1. 创建数据目录
    create_directories()

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头，请检查设备连接。')
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_class = 0      # 当前选中的手势类别
    last_capture_time = 0  # 上次拍摄时间戳（用于冷却）
    saved_count = {c: 0 for c in CLASSES.keys()}  # 本次采集计数

    print(f'\n当前手势类别: [{current_class}] {CLASSES[current_class]}')

    while True:
        # 3. 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print('[警告] 读取摄像头帧失败')
            continue

        # 4. 镜像翻转（更自然，像照镜子）
        frame = cv2.flip(frame, 1)

        # 5. 在画面中央画一个 ROI 方框（引导用户把手放在框内）
        h, w = frame.shape[:2]
        x1 = w // 2 - ROI_SIZE // 2
        y1 = h // 2 - ROI_SIZE // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        # 复制一帧用于显示（不画在原帧上）
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 6. 在左上角显示信息
        cv2.putText(display, f'Class: [{current_class}] {CLASSES[current_class]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f'Captured: {saved_count[current_class]}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 显示总共采集了多少张
        total = sum(saved_count.values())
        cv2.putText(display, f'Total: {total}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 7. 显示画面
        cv2.imshow('Gesture Capture (Press 0-4 to switch, Space to capture, q to quit)',
                   display)

        # 8. 处理按键
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 退出
            break
        elif key in [ord(str(c)) for c in CLASSES.keys()]:
            # 切换手势类别
            current_class = int(chr(key))
            print(f'切换到: [{current_class}] {CLASSES[current_class]}')
        elif key == 32:  # 空格键
            now = time.time()
            if now - last_capture_time < COOLDOWN:
                continue  # 冷却中，忽略
            last_capture_time = now

            # 9. 保存 ROI 区域
            roi = frame[y1:y2, x1:x2].copy()
            filename = get_next_filename(current_class)
            cv2.imwrite(filename, roi)
            saved_count[current_class] += 1
            print(f'[保存] {filename}  (类别累计: {saved_count[current_class]})')

    # 10. 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print('\n采集结束！')
    print('=' * 55)
    for c, name in CLASSES.items():
        folder = os.path.join(DATA_DIR, name)
        count = len([f for f in os.listdir(folder)
                     if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f'  {name}: 共 {count} 张')
    print('=' * 55)


if __name__ == '__main__':
    main()
