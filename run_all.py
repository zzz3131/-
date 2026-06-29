# -*- coding: utf-8 -*-
"""
run_all.py —— 一键运行完整项目流水线
=======================================
使用方法：
    python run_all.py              # 运行完整流水线（合成数据 → 评估）
    python run_all.py --real       # 使用真实数据（需先采集或放入 data/）
    python run_all.py --optimize   # 运行完整流水线 + 优化实验

流水线步骤：
    1. 环境检测
    2. [可选] 生成合成数据
    3. 特征提取 (HOG)
    4. SVM 训练 + 交叉验证
    5. 模型评估（混淆矩阵 + 错误分析）
    6. [可选] 优化实验（LBP+HOG 融合 + 网格搜索）

输出：
    - processed_data/features_hog.npy  # HOG 特征
    - processed_data/labels.npy        # 标签
    - models/svm_gesture.pkl           # 训练好的模型
    - models/scaler.pkl                # 标准化器
    - results/confusion_matrix.png     # 混淆矩阵图
    - results/per_class_accuracy.png   # 各类别准确率
    - results/comparison_results.npz   # 优化对比结果
"""

import sys
import io
import os
import time
import subprocess

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(PROJECT_DIR, 'gesture_env', 'Scripts', 'python.exe')


def run_script(script_name, description):
    """Run a Python script and report success/failure."""
    script_path = os.path.join(PROJECT_DIR, script_name)
    print(f'\n{"#"*60}')
    print(f'#  {description}')
    print(f'#  运行: {script_name}')
    print(f'{"#"*60}')

    start = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=PROJECT_DIR,
        capture_output=False,  # Show output in real-time
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f'\n[OK] {script_name} 完成 (耗时 {elapsed:.1f}s)')
    else:
        print(f'\n[FAIL] {script_name} 失败 (返回码: {result.returncode})')
        return False
    return True


def main():
    real_data = '--real' in sys.argv
    do_optimize = '--optimize' in sys.argv

    print('=' * 60)
    print('  静态手势图像识别系统 —— 完整流水线')
    print('=' * 60)
    print(f'  项目目录: {PROJECT_DIR}')
    print(f'  Python:   {PYTHON}')
    print(f'  使用真实数据: {real_data}')
    print(f'  运行优化实验: {do_optimize}')
    print()

    # ---- Step 1: Environment Check ----
    run_script('check_env.py', 'Step 1/5: 环境检测')

    # ---- Step 2: Data Preparation ----
    if not real_data:
        run_script('generate_synthetic_data.py',
                   'Step 2/5: 生成合成手势数据（测试用）')
        # Copy to processed_data/
        print('\n复制数据到 processed_data/ ...')
        data_dir = os.path.join(PROJECT_DIR, 'data')
        proc_dir = os.path.join(PROJECT_DIR, 'processed_data')
        for class_dir in os.listdir(data_dir):
            src = os.path.join(data_dir, class_dir)
            dst = os.path.join(proc_dir, class_dir)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        import shutil
                        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                count = len([f for f in os.listdir(dst)
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
                print(f'  {class_dir}: {count} 张')
    else:
        print('\n[提示] 使用真实数据，请确保 data/ 下有足够的图像。')
        print('  如果尚未采集，请先运行: python capture_data.py')
        run_script('preprocess.py', 'Step 2/5: 图像预处理 (YCbCr肤色检测)')

    # ---- Step 3: Feature Extraction ----
    run_script('extract_features.py', 'Step 3/5: HOG 特征提取')

    # ---- Step 4: SVM Training ----
    run_script('train_svm.py', 'Step 4/5: SVM 训练 + 交叉验证')

    # ---- Step 5: Evaluation ----
    # Use Agg backend for headless evaluation
    eval_code = """
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
os.chdir(r'{project_dir}')
exec(compile(open('evaluate.py', encoding='utf-8').read(), 'evaluate.py', 'exec'))
""".format(project_dir=PROJECT_DIR)

    print(f'\n{"#"*60}')
    print(f'#  Step 5/5: 模型评估（混淆矩阵 + 错误分析）')
    print(f'{"#"*60}')

    start = time.time()
    result = subprocess.run(
        [PYTHON, '-c', eval_code],
        cwd=PROJECT_DIR,
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f'\n[OK] 评估完成 (耗时 {elapsed:.1f}s)')
    else:
        print(f'\n[FAIL] 评估失败 (返回码: {result.returncode})')

    # ---- Step 6 (Optional): Optimization ----
    if do_optimize:
        opt_code = """
import sys, io, os
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'{project_dir}')
exec(compile(open('optimize.py', encoding='utf-8').read(), 'optimize.py', 'exec'))
""".format(project_dir=PROJECT_DIR)
        print(f'\n{"#"*60}')
        print(f'#  Bonus: 优化实验（LBP+HOG 融合 + 网格搜索）')
        print(f'{"#"*60}')
        subprocess.run([PYTHON, '-c', opt_code], cwd=PROJECT_DIR)

    # ---- Summary ----
    print(f'\n{"="*60}')
    print('  流水线完成！')
    print(f'{"="*60}')
    print('  输出文件：')
    print(f'    特征:    processed_data/features_hog.npy')
    print(f'    标签:    processed_data/labels.npy')
    print(f'    模型:    models/svm_gesture.pkl')
    print(f'    混淆矩阵: results/confusion_matrix.png')
    print(f'    准确率图: results/per_class_accuracy.png')
    if do_optimize:
        print(f'    优化模型: models/svm_optimized.pkl')
        print(f'    对比结果: results/comparison_results.npz')
    print()
    print('  下一步：')
    print('    1. 查看 results/ 下的评估图表')
    print('    2. 用 python capture_data.py 采集真实手势数据')
    print('    3. 用 python demo.py 运行实时手势识别')
    print('    4. 填写 AI_log.md 中的对话记录')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
