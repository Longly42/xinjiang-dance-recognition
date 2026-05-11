# scripts/analyze_dataset.py
"""分析数据集分布，生成详细报告"""

import os
import json
from pathlib import Path
from collections import defaultdict

RAW_VIDEOS_DIR = "data/raw_videos"

# 你的标签映射
STYLE_MAP = {
    "赛乃姆": 0, "刀郎舞": 1, "盘子舞": 2,
    "萨玛舞": 3, "夏地亚纳": 4, "纳孜尔库姆": 5,
}

ACTION_MAP = {
    "扭颈动头": 0, "翻腕绕腕": 1, "摆肩抖肩": 2,
    "垫步进退": 3, "叉腰摆胯": 4, "击掌拍手": 5,
    "旋转": 6, "行礼": 7, "抬臂托帽": 8,
}

def analyze():
    print("=" * 70)
    print("数据集详细分析报告")
    print("=" * 70)
    
    # 统计矩阵
    stats = defaultdict(lambda: defaultdict(int))
    total_videos = 0
    
    for style_name in os.listdir(RAW_VIDEOS_DIR):
        style_path = os.path.join(RAW_VIDEOS_DIR, style_name)
        if not os.path.isdir(style_path):
            continue
        if style_name not in STYLE_MAP:
            continue
        
        for action_name in os.listdir(style_path):
            action_path = os.path.join(style_path, action_name)
            if not os.path.isdir(action_path):
                continue
            if action_name not in ACTION_MAP:
                continue
            
            video_count = len([f for f in Path(action_path).glob("*.mp4")])
            stats[style_name][action_name] = video_count
            total_videos += video_count
    
    # 打印统计表
    print("\n【视频数量分布表】")
    print("-" * 70)
    
    # 表头
    actions = list(ACTION_MAP.keys())
    styles = list(STYLE_MAP.keys())
    
    # 计算每列宽度
    col_width = 12
    header = f"{'风格/动作':<{col_width}}"
    for action in actions:
        header += f"{action:<{col_width}}"
    print(header)
    print("-" * 70)
    
    # 每行数据
    for style in styles:
        row = f"{style:<{col_width}}"
        for action in actions:
            count = stats[style].get(action, 0)
            # 用符号表示有无
            if count > 0:
                display = f"✓ {count}"
            else:
                display = "✗ 0"
            row += f"{display:<{col_width}}"
        print(row)
    
    print("-" * 70)
    print(f"总计视频数: {total_videos}")
    
    # 按动作统计
    print("\n【按动作统计】")
    for action in actions:
        total = sum(stats[style].get(action, 0) for style in styles)
        styles_with = [s for s in styles if stats[s].get(action, 0) > 0]
        print(f"  {action}: {total} 个视频，出现在 {len(styles_with)} 个风格中: {', '.join(styles_with)}")
    
    # 按风格统计
    print("\n【按风格统计】")
    for style in styles:
        total = sum(stats[style].values())
        actions_with = [a for a in actions if stats[style].get(a, 0) > 0]
        print(f"  {style}: {total} 个视频，包含 {len(actions_with)} 种动作: {', '.join(actions_with)}")
    
    # 警告：数据不足的动作
    print("\n【⚠️ 数据量警告】")
    low_action_threshold = 20
    for action in actions:
        total = sum(stats[style].get(action, 0) for style in styles)
        if total < low_action_threshold:
            print(f"  {action}: 只有 {total} 个视频（建议至少 {low_action_threshold} 个）")
    
    # 警告：缺少数据的组合
    print("\n【⚠️ 缺失的组合（该风格缺少该动作）】")
    for style in styles:
        missing = [a for a in actions if stats[style].get(a, 0) == 0]
        if missing:
            print(f"  {style}: 缺少 {', '.join(missing)}")
    
    # 生成建议
    print("\n" + "=" * 70)
    print("【训练建议】")
    print("=" * 70)
    
    # 检查平衡性
    action_totals = {a: sum(stats[s].get(a, 0) for s in styles) for a in actions}
    min_action = min(action_totals.values())
    max_action = max(action_totals.values())
    
    if max_action > min_action * 3:
        print("⚠️ 动作数据分布严重不均衡！")
        print(f"   最多: {max_action} 个，最少: {min_action} 个")
        print("   建议: 增加数据少的动作，或使用类别权重")
    else:
        print("✓ 动作数据分布较均衡")
    
    # 检查风格覆盖
    for action in actions:
        styles_with = [s for s in styles if stats[s].get(action, 0) > 0]
        if len(styles_with) == 1:
            print(f"⚠️ {action} 只出现在 {styles_with[0]} 中")
            print(f"   模型可能无法泛化到其他风格的该动作")
        elif len(styles_with) == 0:
            print(f"⚠️ {action} 没有数据！")

if __name__ == "__main__":
    analyze()