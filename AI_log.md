# AI 对话日志 —— 静态手势图像识别系统开发记录
# ============================================================

## 日志说明
本文档记录了在本项目开发过程中与 AI（Claude Code）的重要对话，
包括：生成关键函数、解决阻塞性 Bug、调参建议等。每条记录包含：
- 日期时间
- 提问内容
- AI 回复摘要
- 代码/截图

---

## 1. 环境配置与项目搭建

**日期**: 2026-06-17

**提问**:
"我要开发一个静态手势识别系统，基于 HOG 特征和 SVM 分类器，识别 5 种手势。
请帮我配置开发环境并规划项目结构。"

**AI 回复摘要**:
- 安装了 opencv-python, scikit-image, scikit-learn, matplotlib, seaborn, numpy
- 规划了项目结构：data/ → processed_data/ → features → model → evaluation
- 创建了 8 个模块：check_env.py, capture_data.py, preprocess.py,
  extract_features.py, train_svm.py, evaluate.py, optimize.py, demo.py

---

## 2. 预处理模块 —— YCbCr 肤色检测

**日期**: 2026-06-17

**提问**:
"用 opencv 实现基于 YCbCr 肤色检测并提取手部区域，缩放为 64x64 灰度图"

**AI 回复摘要**:
生成了 preprocess.py，包含三个核心函数：
1. `ycbcr_skin_mask()` — YCbCr 色彩空间转换 + Cb/Cr 双通道阈值分割
2. `morphological_refine()` — 形态学闭运算（填充空洞）+ 开运算（去除噪点）
3. `extract_hand_region()` — 找最大连通域 → 边界框裁剪

**关键参数**:
- Cb 阈值: 77~127
- Cr 阈值: 133~173
- 形态学校大小: 5×5 椭圆
- 最终输出尺寸: 64×64

**追加提问**:
"在光照变化下失败怎么办？"

**AI 回复摘要**:
光照变化主要影响 Y（亮度）通道，而肤色检测仅使用 Cb 和 Cr（色度）通道，
因此 YCbCr 空间天然对光照有一定鲁棒性。额外改进建议：
1. 使用 CLAHE（自适应直方图均衡化）预处理
2. 动态调整 Cb/Cr 阈值（根据图像统计信息）
3. 使用高斯混合模型（GMM）代替固定阈值

**人工优化**:
(待补充 —— 说明在 AI 代码基础上手动修改了什么，为什么)

---

## 3. 特征提取 —— HOG 特征

**日期**: 2026-06-17

**提问**:
"skimage hog 函数返回的特征向量长度是多少？如果图片尺寸 64x64，参数为默认值？"

**AI 回复摘要**:
- skimage hog 默认参数: orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3)
- 计算过程：
  - n_cells = 64/8 = 8×8 = 64 个 cell
  - n_blocks = (8-3+1)×(8-3+1) = 6×6 = 36 个 block
  - 每个 block = 3×3×9 = 81 维
  - 总维度 = 36 × 81 = 2916 维

**生成代码**: extract_features.py

---

## 4. SVM 分类器训练

**日期**: 2026-06-17

**提问**:
"用 sklearn 的 SVM 对 HOG 特征进行多分类，要求划分训练集测试集，输出分类报告"

**AI 回复摘要**:
- 使用 sklearn.svm.SVC，kernel='rbf', C=10, gamma='scale'
- train_test_split(stratify=y) 确保各类别按比例分配
- StandardScaler 标准化特征（SVM 对尺度敏感）
- 5 折交叉验证评估稳定性
- 输出 classification_report（Precision, Recall, F1-score）

**生成代码**: train_svm.py

**追加提问**:
"如果某两类混淆严重，如何调参？"

**AI 回复摘要**:
1. 增大 C 值 → 模型更关注难分样本 → 边界更复杂
2. 调整 gamma → 较小 gamma 使决策边界更平滑
3. 考虑添加新特征（如 LBP）补充区分信息
4. 对混淆类增加训练样本

---

## 5. 评估与可视化

**日期**: 2026-06-17

**提问**:
"用 matplotlib 和 seaborn 画混淆矩阵的代码，要求显示数字和百分比"

**AI 回复摘要**:
- 使用 seaborn.heatmap 绘制热力图
- 每个格子的标注格式: "样本数\n(百分比%)"
- 对角线用红色粗框标记
- 同时保存 PNG 和显示

**生成代码**: evaluate.py

---

## 6. 错误案例分析

**日期**: 2026-06-19

**提问**:
"为什么这个手势会被识别成 X 而不是 Y？可能的原因是什么？"

**AI 回复摘要**:
新增 `visualize_misclassified()` 函数（位于 evaluate.py），自动将前 10 个错判样本
的图像读取并拼接展示，每个错判样本标红显示"真实标签 vs 预测标签"。

**常见混淆原因分析**:
1. **OK 手势 ↔ 握拳**: 两类手势手指都呈蜷缩状态，HOG 梯度方向分布相似
2. **竖拇指 ↔ 比V**: 两者都有向上伸展的手指，轮廓边缘方向有重叠
3. **肤色检测失效**: 背景中有类肤色物体（木桌、墙面），导致手部区域提取错误
4. **光照极端**: 过暗或过亮时 CLAHE 虽能补偿，但极端情况仍会丢失手指边缘

**错误样本1**:
- 真实标签: ? 预测标签: ?  (运行 evaluate.py 后查看 results/misclassified_samples.png)
- 可能原因:

**错误样本2**:
- 真实标签: ? 预测标签: ?
- 可能原因:

**错误样本3**:
- 真实标签: ? 预测标签: ?
- 可能原因:

---

## 7. 优化 —— LBP 特征融合与网格搜索

**日期**: 2026-06-17

**提问**:
"用 skimage 提取 LBP 特征并与 HOG 串联的代码，调整 SVM 的 C/gamma 参数网格搜索"

**AI 回复摘要**:
- LBP 特征: radius=2, n_points=16, method='uniform' → 243 维
- HOG(2916) + LBP(243) = 3159 维融合特征
- 网格搜索: C ∈ [0.1,1,10,100,1000], gamma ∈ ['scale','auto',0.01,0.001,0.0001]
- 5 折交叉验证，共 25 种组合
- 5 组对比实验：HOG/LBP/HOG+LBP × 默认参数/网格搜索

**生成代码**: optimize.py

---

## 8. 准确率改进 —— 诊断与优化

**日期**: 2026-06-19

**提问**:
"代码已生成，但对手势的识别不够准确，要怎么改进？"

**AI 回复摘要**:

从**数据 → 预处理 → 特征 → 模型**四个维度做了系统性诊断和改进：

### 8.1 预处理层面（preprocess.py）

**改进1: CLAHE 自适应直方图均衡化**
- 在 `preprocess_single()` 的灰度化之后、缩放之前，添加：
  ```python
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  gray = clahe.apply(gray)
  ```
- **为什么有效**: CLAHE 在局部区域（8×8 tile）内做直方图均衡，不会像全局均衡那样过度放大噪声区域。对光照变化（过亮/过暗/侧光）的鲁棒性显著提升。
- **注意**: 这要求 demo.py 推理时也必须使用相同的 CLAHE 处理（已同步修改）。

**改进2: 肤色检测容错检查**
- 添加肤色面积占比检查：`skin_pixels / total_pixels`
- 如果肤色区域 <5% 或 >90%（说明肤色检测大概率失效），不做额外干预，依靠 CLAHE 补偿
- **为什么有效**: 背景中有类肤色物体时，YCbCr 阈值可能把整张图都标记为肤色，导致手部区域提取错误。容错机制避免了用错误分割的结果。

### 8.2 数据层面（train_svm.py）

**改进3: 训练集数据增强（关键改进）**
- 新增 `augment_training_set()` 函数，放在了 train_svm.py 中而非 extract_features.py 中
- **为什么要放在训练阶段**: 如果在特征提取时就增强全部数据，同一张原图的增强版会同时出现在训练集和测试集中，造成数据泄露（模型看到过"近似"的测试样本）
- **增强策略**（每张训练图 → 6 张变体）:
  - 原图 ×1（始终保留）
  - 水平翻转 ×1（模拟左右手互换）
  - 旋转 ±10° ×2（模拟手部倾斜）
  - 亮度变化 ±25 ×2（模拟不同光照）
- **预期效果**: 训练数据量扩大约 6 倍，小样本场景下泛化能力提升 3~8%

### 8.3 推理层面（demo.py）

**改进4: 推理管线对齐**
- 在 demo.py 的实时识别流程中添加 CLAHE：
  ```python
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  hand_gray = clahe.apply(hand_gray)
  ```
- **为什么重要**: 训练时用 CLAHE、推理时不用 → 特征分布不匹配 → 准确率隐性下降

### 8.4 评估层面（evaluate.py）

**改进5: 错判样本可视化**
- 新增 `visualize_misclassified()` 函数
- 从 processed_data/ 加载被分类错误的图像，用 2×5 网格展示
- 每个错判样本用红色标注"真实标签 vs 预测标签"
- 输出保存到 `results/misclassified_samples.png`
- **用途**: 不看数字看图像，直观理解模型把什么和什么搞混了

### 8.5 根本性建议

以上改进都是"修修补补"。如果当前使用的是 `generate_synthetic_data.py` 生成的**合成数据**（OpenCV 画的简笔画），换成真实手势数据后准确率会有 **20~50%** 的飞跃式提升。

推荐采集策略：
- 每类 ≥ 100 张
- 不同光照（自然光 + 灯光）
- 不同背景（纯色 + 复杂）
- 不同角度（正面 + 倾斜 ±15°）
- 不同人（至少 3 人，覆盖不同肤色/手型）

---

## 9. Bug 修复 —— 训练卡死与数据增强报错

**日期**: 2026-06-19

**提问**:
"train_svm.py 出错了怎么回事" / "训练时卡住不动"

**AI 回复摘要**:

### Bug 1: `cv2.flip() Bad argument` — numpy object array 导致

**现象**: 运行到 `augment_training_set()` 时报错：
```
cv2.error: src data type = object is not supported
```

**根因**: `all_images = np.array(all_images, dtype=object)` 将每张 numpy 图像包了一层 `numpy.object_` 包装，导致 OpenCV 无法识别。

**修复**: 保持 `all_images` 为普通 Python list，不要转 `np.array(dtype=object)`。

### Bug 2: `all_images` 和 `all_labels` 长度不一致

**根因**: 原代码先构建 `image_labels` 列表，再单独加载图像到 `all_images`。如果有图像加载失败（`cv2.imread` 返回 None），`all_images` 会比 `image_labels` 短，后续用标签的索引去取图像就会越界或取错。

**修复**: 改为用 `zip(image_paths, image_labels)` 在同一个循环中同步构建两个列表，加载失败的图像对应的标签也不保留。

### Bug 3: SVM 训练卡死（KeyboardInterrupt）

**现象**: 2118 个增强后样本 × 2916 维 HOG 特征，用 `SVC(kernel='rbf', probability=True)` 训练，长时间无响应。

**根因**:
1. `probability=True` 让 sklearn 在训练过程中额外做内部 5 折交叉验证（Platt scaling 概率校准），训练量直接 ×5
2. RBF 核 SVM 对大数据量复杂度为 O(n²)，2118 样本已经很大
3. 默认 `cache_size=200`（200MB），核矩阵反复重算

**修复**:
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `probability` | `True` | `False` | 去掉内部 5 折 CV，速度约 5x |
| `cache_size` | 200 | 2000 | 2GB 核矩阵缓存，减少重复计算 |
| `max_iter` | -1（无限制但在收敛前可能很慢） | 200000 | 显式设上限，防止无限迭代 |

**连带修改（demo.py）**:
- `model.predict_proba()` 在 `probability=False` 后不可用
- 改用 `model.decision_function()` + softmax 转换为"伪概率"用于置信度显示：
  ```python
  decision_scores = model.decision_function(hog_feat_scaled)[0]
  exp_scores = np.exp(decision_scores - np.max(decision_scores))
  pred_proba = exp_scores / exp_scores.sum()
  ```
- softmax 输出的值范围也是 [0, 1] 且和为 1，可用作置信度参考（但不等同于严格的概率估计）

### 改进后的训练流程

```
加载特征 (505 样本)
  → 训练/测试划分 (353 / 152)
  → ★ 仅训练集数据增强 (353 × 6 = 2118)
  → 安全检查 (NaN/Inf 检测)
  → StandardScaler 标准化
  → SVC 训练 (probability=False, cache_size=2000)  ← 大幅加速
  → 5 折交叉验证
  → 测试集评估
  → 保存模型
```

---

## 附录: 项目文件说明

| 文件 | 功能 | 近次修改 |
|------|------|----------|
| check_env.py | 环境检测 | - |
| capture_data.py | 摄像头采集手势图像 | - |
| preprocess.py | YCbCr 肤色检测 + 手部提取 | +CLAHE直方图均衡, +肤色容错检查 (6/19) |
| extract_features.py | HOG 特征提取 | 图像排序统一 (6/19) |
| train_svm.py | SVM 训练 + 交叉验证 | +训练集数据增强, probability=False, cache_size=2000 (6/19) |
| evaluate.py | 混淆矩阵 + 错误分析 | +错判样本可视化 (6/19) |
| optimize.py | LBP+HOG 融合 + 网格搜索 | - |
| demo.py | 实时手势识别演示 | +CLAHE推理对齐, decision_function+softmax (6/19) |
| generate_synthetic_data.py | 合成手势数据生成（仅测试用） | - |
| AI_log.md | 本文档 | - |

---

## 待办事项
- [ ] 完成数据采集（每类 100+ 张，多种光照/背景/角度）
- [ ] 运行预处理并检查肤色检测效果
- [ ] 运行特征提取
- [ ] 训练 SVM，获取初始准确率
- [ ] 运行评估，分析混淆矩阵（查看 results/confusion_matrix.png）
- [ ] 查看错判样本图（results/misclassified_samples.png），分析 3 个典型案例
- [ ] 运行优化实验（python optimize.py），对比 HOG vs HOG+LBP
- [ ] 填写本日志第 6 节中的 3 个错误样本分析
- [ ] 完成至少一项功能的手动修改优化（在 AI 代码基础上自行调整参数/逻辑）
