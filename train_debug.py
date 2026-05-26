"""
check_dataset.py - 检查数据集质量
"""
import numpy as np
import json

# 加载数据
data = np.load('st_data/keypoints/dataset_long.npz')
X = data['X']
y = data['y_style']

# 加载标签
with open('st_data/keypoints/label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)
style_names = [label_map['style'][str(i)] for i in range(6)]

print("=" * 60)
print("数据集质量分析")
print("=" * 60)

# 1. 样本统计
print(f"\n总样本数: {len(X)}")
print(f"样本形状: {X.shape}")
print(f"\n各类别样本数:")
for i, name in enumerate(style_names):
    count = np.sum(y == i)
    print(f"  {name}: {count}")

# 2. 检查是否有全零样本（骨架提取失败）
zero_samples = np.sum(np.sum(X, axis=(1,2,3)) == 0)
print(f"\n全零样本（提取失败）: {zero_samples}")

# 3. 检查每类内部方差（快速检查）
print(f"\n各类别内部方差（帧间差异）:")
for i, name in enumerate(style_names):
    class_samples = X[y == i]
    if len(class_samples) > 1:
        variance = np.var(class_samples)
        print(f"  {name}: {variance:.6f}")

# 4. 类别间距离（快速检查）
print(f"\n类别间平均距离（越小越难区分）:")
for i in range(min(3, len(style_names))):
    for j in range(i+1, min(3, len(style_names))):
        dist = np.mean(np.abs(X[y==i] - X[y==j]))
        print(f"  {style_names[i]} vs {style_names[j]}: {dist:.6f}")

print("\n" + "=" * 60)
print("判断标准:")
print("  - 全零样本 > 10%: 骨架提取有问题")
print("  - 类别间距离 < 0.05: 风格特征不明显")
print("  - 方差过小: 动作变化不足")