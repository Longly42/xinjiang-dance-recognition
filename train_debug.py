"""调试版训练脚本 - 精确定位崩溃位置"""

import os
import sys
import traceback

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU
os.environ["PYTHONVERBOSE"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("调试版训练脚本")
print("=" * 60)

# 1. 加载数据
print("\n[1/8] 加载数据...")
data = np.load('data/keypoints/dataset.npz')
X = data['X']
y_action = data['y_action']
y_style = data['y_style']
print(f"  样本数: {len(X)}")

# 2. 转换为张量
print("\n[2/8] 转换为张量...")
X_tensor = torch.from_numpy(X).float().permute(0, 3, 1, 2)
y_action_tensor = torch.from_numpy(y_action).long()
y_style_tensor = torch.from_numpy(y_style).long()

# 3. 划分数据集
print("\n[3/8] 划分数据集...")
from sklearn.model_selection import train_test_split
indices = list(range(len(X)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

train_X = X_tensor[train_idx]
train_y_a = y_action_tensor[train_idx]
train_y_s = y_style_tensor[train_idx]
val_X = X_tensor[val_idx]
val_y_a = y_action_tensor[val_idx]
val_y_s = y_style_tensor[val_idx]

print(f"  训练集: {len(train_idx)}")
print(f"  验证集: {len(val_idx)}")
print(f"  测试集: {len(test_idx)}")

# 4. 创建 DataLoader
print("\n[4/8] 创建 DataLoader...")
BATCH_SIZE = 8  # 先用小 batch
train_dataset = TensorDataset(train_X, train_y_a, train_y_s)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = TensorDataset(val_X, val_y_a, val_y_s)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"  Batch size: {BATCH_SIZE}")

# 5. 导入模型
print("\n[5/8] 创建模型...")
from model.stgcn import DualTaskSTGCN
model = DualTaskSTGCN()
print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")

# 6. 损失函数和优化器
print("\n[6/8] 设置损失函数和优化器...")
criterion_action = nn.CrossEntropyLoss()
criterion_style = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"  优化器: Adam, lr=0.001")

# 7. 测试一个 batch
print("\n[7/8] 测试前向传播...")
try:
    test_batch_X, test_batch_y_a, test_batch_y_s = next(iter(train_loader))
    action_out, style_out = model(test_batch_X)
    loss_a = criterion_action(action_out, test_batch_y_a)
    loss_s = criterion_style(style_out, test_batch_y_s)
    loss = loss_a + loss_s
    print(f"  前向传播成功, loss: {loss.item():.4f}")
    
    print("  测试反向传播...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  反向传播成功！")
except Exception as e:
    print(f"  失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 8. 训练一个 epoch（带详细日志）
print("\n[8/8] 开始训练一个 epoch...")
print("=" * 60)

model.train()
total_loss = 0
batch_count = 0

for batch_idx, (x, y_a, y_s) in enumerate(train_loader):
    try:
        print(f"  Batch {batch_idx+1}/{len(train_loader)}...", end=' ', flush=True)
        
        optimizer.zero_grad()
        
        action_out, style_out = model(x)
        loss_a = criterion_action(action_out, y_a)
        loss_s = criterion_style(style_out, y_s)
        loss = loss_a + loss_s
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        print(f"Loss: {loss.item():.4f}")
        
        # 每10个batch打印一次统计
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / batch_count
            print(f"  [进度] 已处理 {batch_idx+1}/{len(train_loader)} batches, 平均 Loss: {avg_loss:.4f}")
            
    except Exception as e:
        print(f"\n  ❌ 崩溃在 batch {batch_idx+1}!")
        print(f"  错误: {e}")
        traceback.print_exc()
        
        # 打印出问题的数据
        print(f"\n  问题数据信息:")
        print(f"    x shape: {x.shape}")
        print(f"    x 数值范围: [{x.min():.3f}, {x.max():.3f}]")
        print(f"    y_a: {y_a}")
        print(f"    y_s: {y_s}")
        
        # 检查是否有 NaN
        if torch.isnan(x).any():
            print("    ⚠️ x 包含 NaN!")
        if torch.isinf(x).any():
            print("    ⚠️ x 包含 Inf!")
        
        sys.exit(1)

avg_loss = total_loss / batch_count
print(f"\n✅ 第一个 epoch 完成！平均 Loss: {avg_loss:.4f}")
print("\n训练脚本正常，可以运行完整训练了！")

# 可选：运行更多 epoch
print("\n是否继续训练更多 epoch? (输入 y 继续)")
choice = input().strip().lower()
if choice == 'y':
    print("继续训练...")
    for epoch in range(2, 5):
        print(f"\nEpoch {epoch}/4")
        model.train()
        total_loss = 0
        for batch_idx, (x, y_a, y_s) in enumerate(train_loader):
            optimizer.zero_grad()
            action_out, style_out = model(x)
            loss = criterion_action(action_out, y_a) + criterion_style(style_out, y_s)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} 平均 Loss: {total_loss/len(train_loader):.4f}")
    
    print("\n✅ 训练完成！")