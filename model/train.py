"""
train.py – 训练双任务 ST-GCN，支持早停、学习率调度、模型保存和评估
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（重要！）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER_STEP, LR_SCHEDULER_GAMMA,
    PATIENCE, USE_CLASS_WEIGHTS, MODEL_SAVE_PATH,
    NUM_ACTIONS, NUM_STYLES, KEYPOINTS_DIR
)
from model.dataset import build_dataloaders
from model.stgcn import DualTaskSTGCN


def train_one_epoch(model, loader, optimizer, criterion_action, criterion_style, device, epoch, total_epochs):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_style_loss = 0
    action_acc = 0
    style_acc = 0
    count = 0
    
    num_batches = len(loader)
    print(f"Epoch {epoch}/{total_epochs} [训练] - 共 {num_batches} batches")
    
    for batch_idx, (x, y_action, y_style) in enumerate(loader):
        # 确保标签是 Long 类型
        x = x.to(device)
        y_action = y_action.to(device).long()
        y_style = y_style.to(device).long()
        
        optimizer.zero_grad()

        action_logits, style_logits = model(x)

        loss_a = criterion_action(action_logits, y_action)
        loss_s = criterion_style(style_logits, y_style)
        loss = loss_a + loss_s

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_action_loss += loss_a.item()
        total_style_loss += loss_s.item()

        pred_a = torch.argmax(action_logits, dim=1)
        pred_s = torch.argmax(style_logits, dim=1)
        action_acc += (pred_a == y_action).sum().item()
        style_acc += (pred_s == y_style).sum().item()
        count += x.size(0)
        
        # 每10个batch打印一次进度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            current_acc_a = action_acc / count
            current_acc_s = style_acc / count
            print(f"  Batch {batch_idx+1:3d}/{num_batches} | Loss: {loss.item():.4f} | Acc A: {current_acc_a:.2%} | Acc S: {current_acc_s:.2%}")

    return (total_loss / len(loader),
            total_action_loss / len(loader),
            total_style_loss / len(loader),
            action_acc / count,
            style_acc / count)


def evaluate(model, loader, criterion_action, criterion_style, device, desc="验证"):
    """评估模型"""
    model.eval()
    total_loss = 0
    action_acc = 0
    style_acc = 0
    count = 0

    with torch.no_grad():
        num_batches = len(loader)
        print(f"{desc} - 共 {num_batches} batches")
        
        for batch_idx, (x, y_action, y_style) in enumerate(loader):
            # 确保标签是 Long 类型
            x = x.to(device)
            y_action = y_action.to(device).long()
            y_style = y_style.to(device).long()
            
            action_logits, style_logits = model(x)
            loss = criterion_action(action_logits, y_action) + criterion_style(style_logits, y_style)
            total_loss += loss.item()

            pred_a = torch.argmax(action_logits, dim=1)
            pred_s = torch.argmax(style_logits, dim=1)
            action_acc += (pred_a == y_action).sum().item()
            style_acc += (pred_s == y_style).sum().item()
            count += x.size(0)
            
            # 每5个batch或最后一个batch打印
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Batch {batch_idx+1:3d}/{num_batches}")

    return total_loss / len(loader), action_acc / count, style_acc / count


def get_class_weights(npz_path):
    """从数据集分布计算类别权重"""
    data = np.load(npz_path)
    y_action = data['y_action']
    y_style = data['y_style']

    action_counts = np.bincount(y_action, minlength=NUM_ACTIONS)
    style_counts = np.bincount(y_style, minlength=NUM_STYLES)

    action_weights = 1.0 / (action_counts + 1e-6)
    action_weights = action_weights / action_weights.sum() * NUM_ACTIONS

    style_weights = 1.0 / (style_counts + 1e-6)
    style_weights = style_weights / style_weights.sum() * NUM_STYLES

    return torch.tensor(action_weights, dtype=torch.float), torch.tensor(style_weights, dtype=torch.float)


def save_training_curves(train_losses, val_losses, train_acc_a, val_acc_a, train_acc_s, val_acc_s, save_dir):
    """保存训练曲线图（中文标注）"""
    # 重新设置字体（确保中文正常）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练 Loss', linewidth=2)
    axes[0].plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证 Loss', linewidth=2)
    axes[0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[0].set_ylabel('损失值 (Loss)', fontsize=12)
    axes[0].set_title('训练与验证 Loss 曲线', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(range(1, len(train_acc_a) + 1), train_acc_a, 'b-', label='训练动作准确率', linewidth=2)
    axes[1].plot(range(1, len(val_acc_a) + 1), val_acc_a, 'r-', label='验证动作准确率', linewidth=2)
    axes[1].plot(range(1, len(train_acc_s) + 1), train_acc_s, 'g--', label='训练风格准确率', linewidth=2)
    axes[1].plot(range(1, len(val_acc_s) + 1), val_acc_s, 'm--', label='验证风格准确率', linewidth=2)
    axes[1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
    axes[1].set_ylabel('准确率 (Accuracy)', fontsize=12)
    axes[1].set_title('训练与验证准确率曲线', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 训练曲线已保存: {save_dir}/training_curves.png")


def save_confusion_matrices(model, dataloader, device, save_dir):
    """保存混淆矩阵图（中文标注）"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    model.eval()
    all_action_true = []
    all_action_pred = []
    all_style_true = []
    all_style_pred = []

    print("计算混淆矩阵...")
    with torch.no_grad():
        for batch_idx, (x, y_action, y_style) in enumerate(dataloader):
            x = x.to(device)
            y_action = y_action.to(device).long()
            y_style = y_style.to(device).long()
            
            action_logits, style_logits = model(x)
            action_pred = torch.argmax(action_logits, dim=1)
            style_pred = torch.argmax(style_logits, dim=1)
            all_action_true.extend(y_action.cpu().numpy())
            all_action_pred.extend(action_pred.cpu().numpy())
            all_style_true.extend(y_style.cpu().numpy())
            all_style_pred.extend(style_pred.cpu().numpy())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  处理 batch {batch_idx+1}/{len(dataloader)}")

    # 从 label_map.json 加载实际标签名称
    from model.config import LABEL_MAP_PATH
    
    action_labels = [f"动作{i}" for i in range(NUM_ACTIONS)]
    style_labels = [f"风格{i}" for i in range(NUM_STYLES)]
    
    if os.path.exists(LABEL_MAP_PATH):
        try:
            with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            action_labels = [label_map["action"][str(i)] for i in range(NUM_ACTIONS)]
            style_labels = [label_map["style"][str(i)] for i in range(NUM_STYLES)]
            print(f"  ✅ 加载标签: 动作={action_labels}, 风格={style_labels}")
        except Exception as e:
            print(f"  ⚠️ 加载标签失败: {e}，使用数字标签")
    
    # ==================== 动作混淆矩阵 ====================
    cm_action = confusion_matrix(all_action_true, all_action_pred)
    plt.figure(figsize=(14, 12))
    
    # 使用 seaborn 绘制热力图
    sns.heatmap(cm_action, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=action_labels, 
                yticklabels=action_labels,
                annot_kws={'size': 10})
    
    plt.title('动作分类混淆矩阵', fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_action.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 风格混淆矩阵 ====================
    cm_style = confusion_matrix(all_style_true, all_style_pred)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm_style, 
                annot=True, 
                fmt='d', 
                cmap='Greens',
                xticklabels=style_labels, 
                yticklabels=style_labels,
                annot_kws={'size': 12})
    
    plt.title('风格分类混淆矩阵', fontsize=16, fontweight='bold')
    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_style.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 混淆矩阵已保存: {save_dir}/confusion_matrix_*.png")
    
    # 打印分类报告
    from sklearn.metrics import classification_report
    
    print("\n" + "=" * 60)
    print("动作分类报告:")
    print("=" * 60)
    print(classification_report(all_action_true, all_action_pred, 
                                target_names=action_labels, 
                                digits=4))
    
    print("\n" + "=" * 60)
    print("风格分类报告:")
    print("=" * 60)
    print(classification_report(all_style_true, all_style_pred, 
                                target_names=style_labels, 
                                digits=4))


def train_main():
    """主训练函数"""
    print("=" * 70)
    print("                   新疆舞动作识别模型训练")
    print("=" * 70)
    
    # 设置 matplotlib 中文（全局）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  🔧 使用设备: {device}")
    if device.type == "cuda":
        print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  💾 GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 检查数据集
    npz_path = os.path.join(KEYPOINTS_DIR, "dataset.npz")
    if not os.path.exists(npz_path):
        print(f"\n❌ 错误: 未找到数据集文件 {npz_path}")
        print("   请先运行 scripts/extract_keypoints.py 生成骨架数据")
        return
    
    # 加载数据统计
    data = np.load(npz_path)
    print(f"\n📊 数据集信息:")
    print(f"   样本数: {len(data['X'])}")
    print(f"   序列长度: {data['X'].shape[1]} 帧")
    print(f"   关键点数: {data['X'].shape[2]}")
    print(f"   动作类别数: {NUM_ACTIONS}")
    print(f"   风格类别数: {NUM_STYLES}")
    
    # 类别权重
    if USE_CLASS_WEIGHTS:
        print(f"\n⚖️ 计算类别权重...")
        action_weights, style_weights = get_class_weights(npz_path)
        action_weights = action_weights.to(device)
        style_weights = style_weights.to(device)
        criterion_action = nn.CrossEntropyLoss(weight=action_weights)
        criterion_style = nn.CrossEntropyLoss(weight=style_weights)
        print(f"   动作权重: {action_weights.cpu().numpy().round(2)}")
        print(f"   风格权重: {style_weights.cpu().numpy().round(2)}")
    else:
        criterion_action = nn.CrossEntropyLoss()
        criterion_style = nn.CrossEntropyLoss()

    # 数据加载
    print(f"\n📦 加载数据 (Batch Size: {BATCH_SIZE})...")
    train_loader, val_loader, test_loader = build_dataloaders(npz_path, BATCH_SIZE)
    print(f"   训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    print(f"   验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} batches")
    print(f"   测试集: {len(test_loader.dataset)} 样本, {len(test_loader)} batches")

    # 创建模型
    print(f"\n🏗️ 创建模型...")
    model = DualTaskSTGCN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_SCHEDULER_STEP, gamma=LR_SCHEDULER_GAMMA)
    print(f"   优化器: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"   学习率调度: StepLR (step={LR_SCHEDULER_STEP}, gamma={LR_SCHEDULER_GAMMA})")

    # 训练记录
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_acc_a, val_acc_a = [], []
    train_acc_s, val_acc_s = [], []

    print(f"\n{'='*70}")
    print(f"开始训练 (最大 {EPOCHS} epochs, 早停耐心 {PATIENCE})")
    print(f"{'='*70}\n")

    for epoch in range(1, EPOCHS + 1):
        # 训练
        train_loss, train_loss_a, train_loss_s, train_acc_a_ep, train_acc_s_ep = train_one_epoch(
            model, train_loader, optimizer, criterion_action, criterion_style, device, epoch, EPOCHS)
        
        # 验证
        val_loss, val_acc_a_ep, val_acc_s_ep = evaluate(model, val_loader, criterion_action, criterion_style, device, desc=f"Epoch {epoch} 验证")
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc_a.append(train_acc_a_ep)
        val_acc_a.append(val_acc_a_ep)
        train_acc_s.append(train_acc_s_ep)
        val_acc_s.append(val_acc_s_ep)

        # 打印结果
        print(f"\n{'─'*70}")
        print(f"📊 Epoch {epoch}/{EPOCHS} 结果:")
        print(f"   训练 Loss: {train_loss:.4f} (动作: {train_loss_a:.4f}, 风格: {train_loss_s:.4f})")
        print(f"   训练 Acc:  动作: {train_acc_a_ep:.2%}, 风格: {train_acc_s_ep:.2%}")
        print(f"   验证 Loss: {val_loss:.4f}")
        print(f"   验证 Acc:  动作: {val_acc_a_ep:.2%}, 风格: {val_acc_s_ep:.2%}")
        print(f"   学习率: {current_lr:.6f}")
        print(f"   早停计数: {patience_counter}/{PATIENCE}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ✅ 保存最佳模型 (验证 Loss 降至 {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ 早停触发！训练停止。最佳验证 Loss: {best_val_loss:.4f}")
                break
        
        print(f"{'─'*70}\n")

    # ==================== 测试最佳模型 ====================
    print(f"\n{'='*70}")
    print("测试最佳模型")
    print(f"{'='*70}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"  ✅ 加载最佳模型: {MODEL_SAVE_PATH}")
    else:
        print(f"  ⚠️ 未找到最佳模型，使用最后 epoch 的模型")

    test_loss, test_acc_a, test_acc_s = evaluate(model, test_loader, criterion_action, criterion_style, device, desc="测试集评估")
    
    print(f"\n📈 测试集结果:")
    print(f"   测试 Loss: {test_loss:.4f}")
    print(f"   动作准确率: {test_acc_a:.2%} ({test_acc_a*100:.1f}%)")
    print(f"   风格准确率: {test_acc_s:.2%} ({test_acc_s*100:.1f}%)")

    # ==================== 保存训练曲线 ====================
    print(f"\n📊 生成训练曲线...")
    save_training_curves(train_losses, val_losses, train_acc_a, val_acc_a, train_acc_s, val_acc_s, KEYPOINTS_DIR)

    # ==================== 保存混淆矩阵 ====================
    print(f"📊 生成混淆矩阵...")
    save_confusion_matrices(model, test_loader, device, KEYPOINTS_DIR)

    # ==================== 保存训练日志 ====================
    log_path = os.path.join(KEYPOINTS_DIR, 'training_log.json')
    log_data = {
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_action_acc': test_acc_a,
        'test_style_acc': test_acc_s,
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acc_a': train_acc_a,
        'val_acc_a': val_acc_a,
        'train_acc_s': train_acc_s,
        'val_acc_s': val_acc_s
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  📝 训练日志已保存: {log_path}")

    print(f"\n{'='*70}")
    print("🎉 训练完成！")
    print(f"{'='*70}")
    print(f"  最佳模型: {MODEL_SAVE_PATH}")
    print(f"  测试动作准确率: {test_acc_a:.2%}")
    print(f"  测试风格准确率: {test_acc_s:.2%}")
    print(f"\n  启动 GUI: python main.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        train_main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")