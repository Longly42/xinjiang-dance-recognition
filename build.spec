# -*- mode: python ; coding: utf-8 -*-
"""
build.spec - PyInstaller 打包配置（onedir 模式）

用法:
    pyinstaller build.spec --noconfirm

产物:
    dist/XinjiangDanceRecognition/XinjiangDanceRecognition.exe

说明:
    1. 采用 onedir 而非 onefile。torch + mediapipe + PyQt6 打包后体积在 1GB 上下，
       onefile 每次启动都要解压到临时目录，首次启动会卡十几秒。
    2. 模型权重 best_style_model.pth 会被打进 _internal/st_model/，
       与 st_model/config.py 中基于 __file__ 的路径一致，无需额外处理。
    3. label_map.json 走的是 "./st_data/keypoints" 相对**当前工作目录**的路径，
       无法通过 datas 解决，必须由 build_post.py 复制到 exe 同级目录。
"""

import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

APP_NAME = "XinjiangDanceRecognition"
PROJECT_ROOT = os.path.abspath(SPECPATH)

# ==================== 数据文件 ====================
datas = []

# MediaPipe 的 .binarypb 模型文件、图结构配置等，缺一个都会在运行时崩
datas += collect_data_files("mediapipe")
binaries = collect_dynamic_libs("mediapipe")

# 风格模型权重：必须存在，否则打包后的程序会用随机权重瞎猜
model_path = os.path.join(PROJECT_ROOT, "st_model", "best_style_model.pth")
if os.path.exists(model_path):
    datas.append((model_path, "st_model"))
else:
    print("=" * 68)
    print("[警告] 未找到 st_model/best_style_model.pth")
    print("       打包后的程序将加载随机权重，识别结果无意义。")
    print("       请先运行 python st_model/train.py 生成权重。")
    print("=" * 68)

# 标签映射（同时打进 _internal，作为 label_map 缺失时的兜底备份）
label_map = os.path.join(PROJECT_ROOT, "st_data", "keypoints", "label_map.json")
if os.path.exists(label_map):
    datas.append((label_map, os.path.join("st_data", "keypoints")))

# ==================== 隐式导入 ====================
hiddenimports = []
hiddenimports += collect_submodules("mediapipe")

# scikit-learn 在 PyInstaller 下有若干已知的隐式依赖
hiddenimports += [
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
]

# ==================== 排除项 ====================
# matplotlib / seaborn / pandas 只在训练与绘图阶段使用，推理链路不依赖，
# 排除可显著减小体积。若打包后运行报 matplotlib 相关错误，注释掉本列表即可。
excludes = [
    "matplotlib",
    "seaborn",
    "pandas",
    "tkinter",
    "IPython",
    "notebook",
    "pytest",
    "PyQt5",
]

# ==================== 分析 ====================
a = Analysis(
    [os.path.join(PROJECT_ROOT, "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX 压缩 torch 的 dll 容易导致启动失败，保持关闭
    console=False,      # 无控制台窗口；排障时改为 True 可看到 print 输出
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,          # 有 .ico 时填入路径
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
