import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_path = r"data\to_text\刀郎舞02.mp4"

# 设置图片保存路径
save_dir = r"data\to_text\jpg"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"📁 已创建文件夹: {save_dir}")

cap = cv2.VideoCapture(video_path)

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps > 0:
    duration = total_frames / fps
    print(f"📹 视频时长: {int(duration//60)}分{int(duration%60)}秒 ({duration:.1f}秒)")
    print(f"📊 总帧数: {total_frames}")
    print(f"⚡ 帧率: {fps:.1f} fps")

# 决定采样策略
sample_count = 20  # 总共采样20帧
frames_to_sample = []

if total_frames <= sample_count:
    # 如果视频很短，采样所有帧
    frames_to_sample = list(range(total_frames))
else:
    # 抽取开头、中间、结尾的帧
    start_frames = 5     # 开头5帧
    end_frames = 5       # 结尾5帧
    middle_count = sample_count - start_frames - end_frames  # 中间帧数
    
    # 开头帧
    frames_to_sample.extend(range(min(start_frames, total_frames)))
    
    # 中间均匀采样
    if middle_count > 0:
        middle_start = total_frames // 3  # 从1/3处开始
        middle_end = 2 * total_frames // 3  # 到2/3处结束
        middle_total = middle_end - middle_start
        if middle_total > 0:
            step = max(1, middle_total // middle_count)
            middle_frames = list(range(middle_start, middle_end, step))[:middle_count]
            frames_to_sample.extend(middle_frames)
    
    # 结尾帧
    end_start = max(0, total_frames - end_frames)
    frames_to_sample.extend(range(end_start, total_frames))

print(f"\n🎯 将采样检测 {len(frames_to_sample)} 帧 (开头{start_frames}帧 + 中间{len([f for f in frames_to_sample if f > start_frames and f < total_frames-end_frames])}帧 + 结尾{end_frames}帧)")
print("="*60)

frame_count = 0
detected_count = 0
detected_frames = []  # 记录检测到的帧号

for frame_num in frames_to_sample:
    # 跳到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"帧 {frame_num}: ⚠️  读取失败")
        continue
    
    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # 计算时间位置
    time_pos = frame_num / fps
    
    if results.pose_landmarks:
        detected_count += 1
        detected_frames.append(frame_num)
        
        # 获取关键点坐标
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        print(f"\n✅ 帧 {frame_num} (时间: {time_pos:.1f}秒)")
        print(f"  鼻子: ({nose.x:.2f}, {nose.y:.2f})")
        print(f"  左肩: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f})")
        print(f"  右肩: ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")
        
        # 位置合理性检查
        if left_shoulder.y > nose.y and right_shoulder.y > nose.y:
            print("  ✓ 位置关系合理")
        else:
            print("  ⚠️  位置关系异常")
        
        # 在画面上画出关键点
        h, w, _ = frame.shape
        nose_pixel = (int(nose.x * w), int(nose.y * h))
        left_shoulder_pixel = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_pixel = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        
        cv2.circle(frame, nose_pixel, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_shoulder_pixel, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_shoulder_pixel, 5, (0, 0, 255), -1)
        cv2.line(frame, nose_pixel, left_shoulder_pixel, (255, 0, 0), 2)
        cv2.line(frame, nose_pixel, right_shoulder_pixel, (255, 0, 0), 2)
        
        # 在图片上标注帧号和时间
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time_pos:.1f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存图片
        save_path = os.path.join(save_dir, f"frame_{frame_num}_time_{int(time_pos)}s.jpg")
        cv2.imwrite(save_path, frame)
        print(f"  💾 已保存: {save_path}")
        
    else:
        print(f"❌ 帧 {frame_num} (时间: {time_pos:.1f}秒) - 未检测到人体")

cap.release()
pose.close()

print("\n" + "="*60)
print(f"📊 检测统计:")
print(f"   采样帧数: {frame_count}")
print(f"   检测到人体: {detected_count} 帧")
print(f"   检测率: {detected_count/frame_count*100:.1f}%")
print(f"   检测到的帧位置: {detected_frames}")

# 分析结果
if detected_count == 0:
    print("\n⚠️  警告: 整个视频都没有检测到人体!")
    print("   可能原因:")
    print("   1. 视频中确实没有人")
    print("   2. 人体太小或被遮挡")
    print("   3. 视频质量太差")
elif detected_count < frame_count * 0.5:
    print("\n⚠️  检测率较低，部分时段可能没有人体")
    # 找出连续未检测到的区间
    print("   建议检查未检测到的时段是否有问题")
else:
    print("\n✅ 检测效果良好，人体检测正常")

print(f"\n💡 请检查 {save_dir} 文件夹中的图片验证准确性")