'''
主要代码撰写位置，其余位置不建议更改
'''
import cv2
import numpy as np

#######################################################################################
class Solution:
    def __init__(self):
        self.base_speed = 20.0      # 基础速度
        self.kp = 0.036            # 比例系数
        self.kd = 0.001            # 微分系数
        self.last_error = 0
        self.road_width = 320 
        pass
    def run(self,carImg,imu):
        """
        输入宽320高240彩色RGB图像carImg，六轴传感器imu数据前三个加速度，后三个角速度，
        输出严格为两个数字代表左右轮子转速
        """
        img = cv2.cvtColor(carImg, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # ROI：下半部分
        roi = img[h // 2:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        H, W = binary.shape
        centers = []

        # 从下往上逐行扫描
        for row in range(H - 1, -1, -1):
            cols = np.where(binary[row] > 0)[0]
            if len(cols) < 10:          # 这一行几乎没白点，跳过
                continue

            left_edge = cols[0]
            right_edge = cols[-1]

            # 判断边线是否“贴到图像边缘” → 视为缺失
            left_missing = left_edge <= 3
            right_missing = right_edge >= W - 4

            if left_missing and right_missing:
                # 两侧都缺失：无法判断，跳过（用其他行的结果）
                continue
            elif left_missing:
                # 只有右边线可见 → 左侧被切掉 → 用估计路宽补全左侧
                left_edge = right_edge - self.road_width
            elif right_missing:
                # 只有左边线可见 → 右侧被切掉 → 补全右侧
                right_edge = left_edge + self.road_width
            # else: 两侧都可见，正常

            center = (left_edge + right_edge) // 2
            centers.append(center)

        if centers:
            # 距离车最近的几行权重更大
            weights = np.linspace(1.5, 1.0, len(centers))
            center_x = int(np.average(centers, weights=weights))
        else:
            center_x = W // 2           # 整帧都没白点，直行

        # 误差 & PD 控制
        error = center_x - W // 2
        d_error = error - self.last_error
        self.last_error = error
        steer = self.kp * error + self.kd * d_error

        # 弯道减速
        speed = self.base_speed - abs(error) * 0.003
        speed = float(np.clip(speed, 0.8, 2.0))

        leftspeed = float(np.clip(speed + steer, -50.0, 50.0))
        rightspeed = float(np.clip(speed - steer, -50.0, 50.0))

        cv2.imshow('binary', binary)
        cv2.imshow('carimg', img)
        cv2.waitKey(1)

        return leftspeed, rightspeed

#######################################################################################
"""
manual参数切换手动自动控制
"""
if __name__=="__main__":
    from sim import run
    solution=Solution()
    run(solution.run,manual=False)