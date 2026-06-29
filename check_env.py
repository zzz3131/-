# -*- coding: utf-8 -*-
"""
Environment check script -- verify all dependencies for this project
Usage:
    python check_env.py
"""
import sys
import io

# Force UTF-8 output on Windows to avoid GBK encoding errors
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_package(name, import_name=None):
    """
    Check if a Python package is installed.
    :param name:        pip package name
    :param import_name: import name (defaults to `name`)
    """
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'[OK] {name:25s} installed, version: {version}')
        return True
    except ImportError:
        print(f'[!!] {name:25s} NOT installed. Run: pip install {name}')
        return False


def main():
    print('=' * 55)
    print('  Hand Gesture Recognition -- Environment Check')
    print('=' * 55)
    print(f'Python version: {sys.version}')
    print('-' * 55)

    all_ok = True
    all_ok &= check_package('numpy')
    all_ok &= check_package('opencv-python',    'cv2')
    all_ok &= check_package('scikit-image',     'skimage')
    all_ok &= check_package('scikit-learn',     'sklearn')
    all_ok &= check_package('matplotlib')
    all_ok &= check_package('seaborn')
    all_ok &= check_package('joblib')

    print('-' * 55)
    if all_ok:
        print('OK -- All dependencies ready.')
    else:
        print('ERROR -- Missing dependencies. See above.')
    print('=' * 55)


if __name__ == '__main__':
    main()
