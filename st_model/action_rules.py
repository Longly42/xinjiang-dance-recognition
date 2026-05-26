"""
action_rules.py - 基于骨架关键点运动轨迹的规则动作识别（无需训练）
识别新疆舞常见细粒度动作：扭颈、翻腕、摆肩、垫步、叉腰摆胯、击掌、旋转、行礼、抬臂托帽
"""

import numpy as np
from collections import deque
import math


class RuleBasedActionRecognizer:
    def __init__(self, history_frames=15):
        """
        history_frames: 用于计算运动特征的滑动窗口长度（帧数）
        """
        self.history_frames = history_frames
        # 存储历史关键点轨迹 (x, y, visibility)
        self.pose_history = deque(maxlen=history_frames)
        # 上次动作结果（用于平滑）
        self.last_action = "无动作"
        self.smooth_counter = 0

    def update(self, pose):
        """
        更新最新一帧的骨架 (33, 3)
        调用 get_action() 获取当前帧识别出的动作
        """
        self.pose_history.append(pose.copy())

    def get_action(self):
        """
        基于历史窗口内的骨架序列，识别当前最可能的动作
        返回动作名称（字符串）
        """
        if len(self.pose_history) < 5:
            return "无动作"

        # 分别计算各个动作的得分
        scores = {}
        scores["扭颈动头"] = self._head_turn_score()
        scores["翻腕绕腕"] = self._wrist_rotation_score()
        scores["摆肩抖肩"] = self._shoulder_shake_score()
        scores["垫步进退"] = self._step_move_score()
        scores["叉腰摆胯"] = self._hip_sway_score()
        scores["击掌拍手"] = self._clap_score()
        scores["旋转"] = self._spin_score()
        scores["行礼"] = self._bow_score()
        scores["抬臂托帽"] = self._arm_raise_score()

        # 得分最高的动作（且大于阈值）
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        if best_score < 0.3:   # 阈值，低于此值认为无动作
            best_action = "无动作"

        # 简单平滑：如果与上一帧相同则增加置信，否则可能跳变
        if best_action == self.last_action:
            self.smooth_counter = min(3, self.smooth_counter + 1)
        else:
            self.smooth_counter = max(0, self.smooth_counter - 1)
        if self.smooth_counter >= 2:
            return best_action
        else:
            # 如果新动作连续出现2帧才切换
            return self.last_action if self.last_action != "无动作" else best_action

    # ------------------ 各个动作的评分函数 ------------------
    def _head_turn_score(self):
        """头部转动：利用鼻子(0)和肩膀中心(11,12中点)的连线角度变化"""
        if len(self.pose_history) < 3:
            return 0.0
        # 获取最近3帧的头部偏转角
        angles = []
        for pose in list(self.pose_history)[-3:]:
            nose = pose[0]
            left_shoulder = pose[11]
            right_shoulder = pose[12]
            if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
                continue
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                               (left_shoulder[1] + right_shoulder[1]) / 2)
            # 向量：肩中心到鼻子
            dx = nose[0] - shoulder_center[0]
            dy = nose[1] - shoulder_center[1]
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
        if len(angles) < 2:
            return 0.0
        # 角度变化幅度（最大-最小）
        angle_change = max(angles) - min(angles)
        score = min(1.0, angle_change / 30.0)   # 30度以上认为是转头
        return score

    def _wrist_rotation_score(self):
        """翻腕：手腕(15,16)相对于肘(13,14)的旋转角速度"""
        if len(self.pose_history) < 4:
            return 0.0
        # 取左腕和右腕分别计算
        left_wrist_angles = []
        right_wrist_angles = []
        poses = list(self.pose_history)
        for i in range(len(poses)-1):
            p1 = poses[i]
            p2 = poses[i+1]
            # 左臂：肘(13) -> 腕(15)
            if p1[13][2] > 0.5 and p1[15][2] > 0.5 and p2[15][2] > 0.5:
                v1 = (p1[15][0] - p1[13][0], p1[15][1] - p1[13][1])
                v2 = (p2[15][0] - p1[13][0], p2[15][1] - p1[13][1])
                ang1 = math.atan2(v1[1], v1[0])
                ang2 = math.atan2(v2[1], v2[0])
                left_wrist_angles.append(abs(ang2 - ang1))
            # 右臂类似
            if p1[14][2] > 0.5 and p1[16][2] > 0.5 and p2[16][2] > 0.5:
                v1 = (p1[16][0] - p1[14][0], p1[16][1] - p1[14][1])
                v2 = (p2[16][0] - p1[14][0], p2[16][1] - p1[14][1])
                ang1 = math.atan2(v1[1], v1[0])
                ang2 = math.atan2(v2[1], v2[0])
                right_wrist_angles.append(abs(ang2 - ang1))
        all_deltas = left_wrist_angles + right_wrist_angles
        if not all_deltas:
            return 0.0
        max_angular_speed = max(all_deltas)
        score = min(1.0, max_angular_speed / 1.0)   # 弧度每秒，阈值1.0 rad/s
        return score

    def _shoulder_shake_score(self):
        """摆肩抖肩：左右肩垂直方向位移的频率和幅度"""
        if len(self.pose_history) < 10:
            return 0.0
        left_y = []
        right_y = []
        for pose in self.pose_history:
            if pose[11][2] > 0.5:
                left_y.append(pose[11][1])
            if pose[12][2] > 0.5:
                right_y.append(pose[12][1])
        if len(left_y) < 5 or len(right_y) < 5:
            return 0.0
        # 计算垂直方向的标准差（抖动幅度）
        left_std = np.std(left_y[-10:])
        right_std = np.std(right_y[-10:])
        avg_std = (left_std + right_std) / 2
        # 幅度超过0.02（归一化坐标）认为是抖肩
        score = min(1.0, avg_std / 0.02)
        # 同时检查左右交替性（简化：左肩和右肩的相位差）
        # 可通过零交叉频率粗略判断，这里略
        return score

    def _step_move_score(self):
        """垫步进退：脚踝的水平位移速度"""
        if len(self.pose_history) < 4:
            return 0.0
        left_ankle = []
        right_ankle = []
        for pose in self.pose_history:
            if pose[27][2] > 0.5:
                left_ankle.append(pose[27][0])
            if pose[28][2] > 0.5:
                right_ankle.append(pose[28][0])
        if len(left_ankle) < 2 or len(right_ankle) < 2:
            return 0.0
        left_speed = abs(left_ankle[-1] - left_ankle[-2]) / (1.0 / 30)  # 假设30fps
        right_speed = abs(right_ankle[-1] - right_ankle[-2]) / (1.0 / 30)
        avg_speed = (left_speed + right_speed) / 2
        score = min(1.0, avg_speed / 0.5)   # 0.5 归一化坐标/秒 为阈值
        return score

    def _hip_sway_score(self):
        """叉腰摆胯：手腕靠近髋部 + 髋部左右摆动"""
        # 先判断手腕是否在髋部附近
        wrist_near_hip = False
        pose = self.pose_history[-1]
        if pose[15][2] > 0.5 and pose[23][2] > 0.5:
            dist = math.hypot(pose[15][0] - pose[23][0], pose[15][1] - pose[23][1])
            if dist < 0.15:
                wrist_near_hip = True
        if not wrist_near_hip:
            return 0.0
        # 髋部左右摆动幅度
        hip_x = []
        for p in self.pose_history:
            if p[23][2] > 0.5:
                hip_x.append(p[23][0])
        if len(hip_x) < 5:
            return 0.0
        swing = np.std(hip_x[-10:])
        score = min(1.0, swing / 0.03)
        return score

    def _clap_score(self):
        """击掌拍手：两手距离快速减小到接近0"""
        if len(self.pose_history) < 2:
            return 0.0
        last = self.pose_history[-1]
        prev = self.pose_history[-2]
        if last[15][2] < 0.5 or last[16][2] < 0.5 or prev[15][2] < 0.5 or prev[16][2] < 0.5:
            return 0.0
        dist_last = math.hypot(last[15][0] - last[16][0], last[15][1] - last[16][1])
        dist_prev = math.hypot(prev[15][0] - prev[16][0], prev[15][1] - prev[16][1])
        if dist_last < 0.05 and dist_prev > 0.1:
            return 1.0
        return 0.0

    def _spin_score(self):
        """旋转：身体朝向角度的累积变化（通过肩膀连线角度）"""
        if len(self.pose_history) < 10:
            return 0.0
        angles = []
        for pose in self.pose_history:
            if pose[11][2] > 0.5 and pose[12][2] > 0.5:
                dx = pose[12][0] - pose[11][0]
                dy = pose[12][1] - pose[11][1]
                angle = math.degrees(math.atan2(dy, dx))
                angles.append(angle)
        if len(angles) < 3:
            return 0.0
        # 总旋转角度（考虑跨越360度）
        total_rotation = 0
        for i in range(1, len(angles)):
            delta = angles[i] - angles[i-1]
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            total_rotation += abs(delta)
        score = min(1.0, total_rotation / 180.0)   # 旋转超过180度认为旋转动作
        return score

    def _bow_score(self):
        """行礼：上半身前倾角度（躯干与垂直轴夹角）"""
        pose = self.pose_history[-1]
        if pose[11][2] < 0.5 or pose[23][2] < 0.5:
            return 0.0
        # 肩中点
        shoulder_y = (pose[11][1] + pose[12][1]) / 2
        hip_y = (pose[23][1] + pose[24][1]) / 2
        # 前倾：肩y坐标大于髋y坐标（低头）
        lean = shoulder_y - hip_y
        # 归一化：lean > 0.1 认为是行礼
        score = min(1.0, max(0.0, lean / 0.1))
        return score

    def _arm_raise_score(self):
        """抬臂托帽：手臂上举，手腕在头部上方"""
        pose = self.pose_history[-1]
        head_y = pose[0][1]   # 鼻子y
        left_wrist_y = pose[15][1] if pose[15][2] > 0.5 else None
        right_wrist_y = pose[16][1] if pose[16][2] > 0.5 else None
        if left_wrist_y is None and right_wrist_y is None:
            return 0.0
        wrist_up = False
        if left_wrist_y is not None and left_wrist_y < head_y - 0.05:
            wrist_up = True
        if right_wrist_y is not None and right_wrist_y < head_y - 0.05:
            wrist_up = True
        if wrist_up:
            # 检查手臂是否伸直（肘与腕距离）
            if left_wrist_y is not None and pose[13][2] > 0.5:
                arm_len = math.hypot(pose[15][0]-pose[13][0], pose[15][1]-pose[13][1])
                if arm_len > 0.2:
                    return 1.0
            if right_wrist_y is not None and pose[14][2] > 0.5:
                arm_len = math.hypot(pose[16][0]-pose[14][0], pose[16][1]-pose[14][1])
                if arm_len > 0.2:
                    return 1.0
        return 0.0