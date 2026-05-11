import os

# 基础目录
base = "raw_videos"

# 定义：风格 -> 动作列表
folders = {
    "赛乃姆": ["扭颈动头", "翻腕绕腕", "摆肩抖肩", "垫步进退", "抬臂托帽", "旋转"],
    "刀郎舞": ["摆肩抖肩", "垫步进退", "叉腰摆胯", "击掌拍手", "行礼"],
    "盘子舞": ["翻腕绕腕", "抬臂托帽", "垫步进退", "旋转"],
    "萨玛舞": ["摆肩抖肩", "垫步进退", "旋转", "行礼"],
    "夏地亚纳": ["抬臂托帽", "垫步进退", "旋转", "击掌拍手"],
    "纳孜尔库姆": ["扭颈动头", "翻腕绕腕", "垫步进退", "叉腰摆胯"]
}

# 自动创建
for style, actions in folders.items():
    for act in actions:
        path = os.path.join(base, style, act)
        os.makedirs(path, exist_ok=True)

print("✅ 所有文件夹已一键生成完成！")