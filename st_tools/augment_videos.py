"""
augment_videos.py - 视频数据增强工具
对 st_data/raw_videos 中的视频进行多种增强，生成新视频
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ==================== 配置 ====================
INPUT_DIR = "st_data/raw_videos"
OUTPUT_DIR = "st_data/raw_videos_augmented"  # 增强后的视频输出目录

# 增强策略配置
AUGMENTATION_CONFIG = {
    # 几何变换
    'horizontal_flip': True,      # 水平翻转
    'vertical_flip': False,        # 垂直翻转（一般不适用于舞蹈）
    'rotate': [5, -5, 10, -10],    # 旋转角度（度）
    'scale': [0.9, 1.1],           # 缩放比例
    
    # 色彩变换（对骨架提取影响较小，但可以增加鲁棒性）
    'brightness': [0.8, 1.2],      # 亮度调整
    'contrast': [0.8, 1.2],        # 对比度调整
    
    # 时序变换
    'time_reverse': True,           # 时间反转（倒放）
    'time_mask': True,              # 时间掩码（随机遮挡若干帧）
    'time_stretch': [0.9, 1.1],     # 时间拉伸
    
    # 噪声
    'gaussian_noise': True,         # 高斯噪声
    'motion_blur': True,            # 运动模糊
}

# 每个原始视频生成的增强版本数
AUGMENTATIONS_PER_VIDEO = 5  # 建议 3-5


def horizontal_flip_video(input_path, output_path):
    """水平翻转视频"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 1)  # 水平翻转
        out.write(flipped)
    
    cap.release()
    out.release()


def rotate_video(input_path, output_path, angle):
    """旋转视频"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算旋转后的画布大小
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated = cv2.warpAffine(frame, matrix, (new_width, new_height))
        out.write(rotated)
    
    cap.release()
    out.release()


def scale_video(input_path, output_path, scale_factor):
    """缩放视频"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        scaled = cv2.resize(frame, (new_width, new_height))
        out.write(scaled)
    
    cap.release()
    out.release()


def adjust_brightness_contrast(input_path, output_path, brightness, contrast):
    """调整亮度和对比度"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        out.write(adjusted)
    
    cap.release()
    out.release()


def time_reverse_video(input_path, output_path):
    """时间反转（倒放）"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读取所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 反转帧顺序
    frames.reverse()
    
    # 写入反转后的视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def add_gaussian_noise(input_path, output_path, intensity=25):
    """添加高斯噪声"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        noise = np.random.normal(0, intensity, frame.shape).astype(np.uint8)
        noisy = cv2.add(frame, noise)
        out.write(noisy)
    
    cap.release()
    out.release()


def add_motion_blur(input_path, output_path, kernel_size=5):
    """添加运动模糊"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = cv2.filter2D(frame, -1, kernel)
        out.write(blurred)
    
    cap.release()
    out.release()


def time_mask_video(input_path, output_path, mask_ratio=0.1):
    """时间掩码：随机遮挡连续若干帧（用前一帧填充）"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读取所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 随机选择掩码区间
    total_frames = len(frames)
    mask_length = max(1, int(total_frames * mask_ratio))
    mask_start = random.randint(0, total_frames - mask_length)
    
    # 用前一帧填充掩码区间
    for i in range(mask_start, mask_start + mask_length):
        if i > 0:
            frames[i] = frames[i-1].copy()
    
    # 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def time_stretch_video(input_path, output_path, stretch_factor):
    """时间拉伸（改变播放速度）"""
    cap = cv2.VideoCapture(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 读取所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 调整帧率实现时间拉伸
    new_fps = orig_fps / stretch_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    # 如果需要拉伸到更多帧，重复帧
    if stretch_factor > 1:
        # 慢放：重复帧
        new_frames = []
        for frame in frames:
            repeat = int(stretch_factor)
            for _ in range(repeat):
                new_frames.append(frame)
        frames = new_frames[:int(len(frames) * stretch_factor)]
    elif stretch_factor < 1:
        # 快放：跳帧
        step = int(1 / stretch_factor)
        frames = frames[::step]
    
    for frame in frames:
        out.write(frame)
    out.release()


def augment_single_video(video_path, output_dir, style_name):
    """对单个视频应用多种增强"""
    base_name = Path(video_path).stem
    ext = Path(video_path).suffix
    
    generated = []
    aug_idx = 0
    
    # 1. 水平翻转（几乎总是有效）
    if AUGMENTATION_CONFIG['horizontal_flip']:
        out_path = os.path.join(output_dir, f"{base_name}_flip{aug_idx}{ext}")
        horizontal_flip_video(video_path, out_path)
        generated.append(out_path)
        aug_idx += 1
    
    # 2. 旋转增强
    for angle in AUGMENTATION_CONFIG['rotate']:
        out_path = os.path.join(output_dir, f"{base_name}_rot{angle}{ext}")
        rotate_video(video_path, out_path, angle)
        generated.append(out_path)
        aug_idx += 1
        if aug_idx >= AUGMENTATIONS_PER_VIDEO:
            break
    
    # 3. 时间反转
    if AUGMENTATION_CONFIG['time_reverse'] and aug_idx < AUGMENTATIONS_PER_VIDEO:
        out_path = os.path.join(output_dir, f"{base_name}_reverse{ext}")
        time_reverse_video(video_path, out_path)
        generated.append(out_path)
        aug_idx += 1
    
    # 4. 时间掩码
    if AUGMENTATION_CONFIG['time_mask'] and aug_idx < AUGMENTATIONS_PER_VIDEO:
        out_path = os.path.join(output_dir, f"{base_name}_timemask{ext}")
        time_mask_video(video_path, out_path, mask_ratio=0.15)
        generated.append(out_path)
        aug_idx += 1
    
    # 5. 噪声增强
    if AUGMENTATION_CONFIG['gaussian_noise'] and aug_idx < AUGMENTATIONS_PER_VIDEO:
        out_path = os.path.join(output_dir, f"{base_name}_noise{ext}")
        add_gaussian_noise(video_path, out_path, intensity=15)
        generated.append(out_path)
        aug_idx += 1
    
    # 6. 亮度/对比度调整
    if aug_idx < AUGMENTATIONS_PER_VIDEO:
        brightness = random.uniform(*AUGMENTATION_CONFIG['brightness'])
        contrast = random.uniform(*AUGMENTATION_CONFIG['contrast'])
        out_path = os.path.join(output_dir, f"{base_name}_bright{ext}")
        adjust_brightness_contrast(video_path, out_path, 
                                    int(50 * (brightness - 1)), contrast)
        generated.append(out_path)
        aug_idx += 1
    
    return generated


def main():
    """批量增强所有视频"""
    print("=" * 60)
    print("视频数据增强工具")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每个视频生成: {AUGMENTATIONS_PER_VIDEO} 个增强版本")
    print("=" * 60)
    
    # 创建输出目录（保持相同结构）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 统计
    total_original = 0
    total_augmented = 0
    
    # 遍历每个风格
    for style_name in os.listdir(INPUT_DIR):
        style_path = os.path.join(INPUT_DIR, style_name)
        if not os.path.isdir(style_path):
            continue
        
        print(f"\n📁 处理风格: {style_name}")
        
        # 创建输出目录
        output_style_dir = os.path.join(OUTPUT_DIR, style_name)
        os.makedirs(output_style_dir, exist_ok=True)
        
        # 获取视频文件
        video_files = [f for f in os.listdir(style_path) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            print(f"  没有视频文件")
            continue
        
        print(f"  找到 {len(video_files)} 个原始视频")
        
        # 复制原始视频到输出目录
        for video_file in video_files:
            src_path = os.path.join(style_path, video_file)
            dst_path = os.path.join(output_style_dir, video_file)
            if not os.path.exists(dst_path):
                import shutil
                shutil.copy2(src_path, dst_path)
            total_original += 1
        
        # 生成增强视频
        for video_file in tqdm(video_files, desc=f"  增强中"):
            src_path = os.path.join(style_path, video_file)
            augmented = augment_single_video(src_path, output_style_dir, style_name)
            total_augmented += len(augmented)
    
    # 统计报告
    print("\n" + "=" * 60)
    print("增强完成！")
    print("=" * 60)
    print(f"原始视频数: {total_original}")
    print(f"增强视频数: {total_augmented}")
    print(f"总视频数: {total_original + total_augmented}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n下一步:")
    print(f"  1. 将增强视频合并到原目录")
    print(f"  2. 或直接使用增强目录进行骨架提取")
    print("=" * 60)


if __name__ == "__main__":
    main()