# 静态手势识别系统

基于计算机视觉和机器学习的实时手势识别项目，使用 HOG 特征提取 + SVM 分类器实现。

## 项目功能

- **实时手势识别**：通过摄像头实时识别5种手势
  - ✊ 握拳 (Fist)
  - ✋ 伸掌 (Palm)
  - 👍 竖拇指 (Thumb)
  - 👌 OK
  - ✌️ 比V (Victory)

- **完整 ML 流水线**：数据采集 → 预处理 → 特征提取 → 模型训练 → 评估 → 实时预测

## 技术栈

| 技术 | 用途 |
|------|------|
| Python | 主要开发语言 |
| OpenCV | 图像处理、摄像头交互 |
| scikit-learn | SVM 分类器、模型评估 |
| scikit-image | HOG 特征提取 |
| NumPy | 数值计算 |
| Matplotlib | 数据可视化 |

## 核心算法

1. **肤色检测**：YCbCr 色彩空间阈值分割
2. **形态学处理**：闭运算填充空洞 + 开运算去噪
3. **特征提取**：HOG（方向梯度直方图）特征
4. **分类器**：SVM 支持向量机（RBF 核）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整流水线

```bash
# 使用合成数据快速体验
python run_all.py

# 或使用真实数据（需先采集）
python run_all.py --real
```

### 3. 实时演示

```bash
python demo.py
```

按 `q` 退出，按 `s` 截图保存。

## 项目结构

```
├── capture_data.py          # 数据采集模块
├── preprocess.py            # 图像预处理（肤色检测）
├── extract_features.py      # HOG 特征提取
├── train_svm.py            # SVM 模型训练
├── evaluate.py             # 模型评估与可视化
├── demo.py                 # 实时手势识别演示
├── run_all.py              # 一键运行完整流水线
├── optimize.py             # 超参数优化
├── check_env.py            # 环境检测
└── generate_synthetic_data.py  # 合成数据生成
```

## 实现亮点

- **数据增强**：翻转、旋转、亮度变化，提升模型泛化能力
- **交叉验证**：5折交叉验证确保模型稳定性
- **滑动窗口平滑**：预测结果时序平滑，减少抖动
- **中文注释完整**：每个模块都有详细的技术说明

## 模型性能

在测试集上典型表现：
- 准确率：> 90%
- 支持混淆矩阵可视化分析
- 支持错误案例自动展示

## 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 摄像头（用于实时演示）

## 许可证

MIT License
