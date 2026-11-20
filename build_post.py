# -*- mode: python ; coding: utf-8 -*-
"""
build_post.py - 打包后处理

用法:
    pyinstaller build.spec --noconfirm && python build_post.py

解决的问题:
    utils/config.py 与 st_model/config.py 中，标签映射用的是
        LABEL_MAP_PATH = "./st_data/keypoints/label_map.json"
    即相对**当前工作目录**的路径。打包成 exe 后双击运行，工作目录是 exe 所在目录，
    若该处没有 st_data/keypoints/label_map.json，程序不会报错，而是静默退化为
    ["风格0", "风格1", ...] 这样无意义的标签名。

    本脚本把标签映射复制到 exe 同级，并做一次完整性自检。
"""

import os
import shutil
import sys

APP_NAME = "XinjiangDanceRecognition"
DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", APP_NAME)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def copy_label_map():
    src = os.path.join(PROJECT_ROOT, "st_data", "keypoints", "label_map.json")
    dst_dir = os.path.join(DIST_DIR, "st_data", "keypoints")
    dst = os.path.join(dst_dir, "label_map.json")

    if not os.path.exists(src):
        print(f"[跳过] 源标签映射不存在: {src}")
        return False

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[完成] 标签映射 -> {dst}")
    return True


def verify():
    """检查打包产物关键文件是否齐全"""
    exe_path = os.path.join(DIST_DIR, f"{APP_NAME}.exe")
    checks = [
        ("主程序", exe_path),
        ("标签映射", os.path.join(DIST_DIR, "st_data", "keypoints", "label_map.json")),
        ("风格权重", os.path.join(DIST_DIR, "_internal", "st_model", "best_style_model.pth")),
        ("MediaPipe 数据", os.path.join(DIST_DIR, "_internal", "mediapipe")),
    ]

    print("\n" + "=" * 60)
    print("打包产物自检")
    print("=" * 60)

    all_ok = True
    for name, path in checks:
        ok = os.path.exists(path)
        mark = "OK  " if ok else "缺失"
        print(f"  [{mark}] {name:14} {path if ok else '(未找到)'}")
        if not ok:
            all_ok = False

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / 1024 / 1024
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(DIST_DIR)
            for f in fs
        ) / 1024 / 1024
        print(f"\n  主程序体积: {size_mb:.1f} MB")
        print(f"  整体目录:   {total:.0f} MB")

    if all_ok:
        print("\n  自检通过。将整个目录压缩后即可分发：")
        print(f"  {DIST_DIR}")
    else:
        print("\n  自检未通过：缺权重请先训练；缺 MediaPipe 数据请检查 spec 中的数据收集。")

    return all_ok


if __name__ == "__main__":
    if not os.path.isdir(DIST_DIR):
        print(f"错误: 未找到打包产物目录 {DIST_DIR}")
        print("请先执行: pyinstaller build.spec --noconfirm")
        sys.exit(1)

    copy_label_map()
    sys.exit(0 if verify() else 1)
