import time
import os
import pybullet as p
from math import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import heightfield

L1 = 0.054
L2 = 0.0661
L3 = 0.08
model_name = "2026-09-09 17_40_30"

class Hexapod:
    """
    #参数顺序a{i-1},alpha{i-1},d{i},theta{i}
    DH_list = [[0,0,0,theta1],
               [0.054,pi/2,0,-theta2],
               [0.0661,0,0,theta3-pi/4],
               [0.08,0,0,0]]
    """
    def __init__(self,L1,L2,L3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.target = random.uniform(-pi/2, pi/2)
        self.init_force = 2
        self.hexapod = p.loadURDF(r"./phantomx_description/urdf/phantomx.urdf", [0, 0, 0.5],p.getQuaternionFromEuler([0, 0, 0]))
        """
        将关节id存为字典-列表形式
        rf表示右前腿，rm表示右中腿，rr表示右后腿
        lf表示左前腿，lm表示左中腿，lr表示左后腿
        l[0]表示第一个关节，l[1]表示第二个关节,l[2]表示第三个关节
        """
        self.joint_name_to_id = {"rf": [1, 3, 4], "rm": [5, 7, 8], "rr": [9, 11, 12], "lf": [13, 15, 16], "lm": [17, 19, 20],"lr": [21, 23, 24]}
        for name in self.joint_name_to_id:
            p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][0],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0, force=self.init_force)
            p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][1],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0, force=self.init_force)
            p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][2],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0, force=self.init_force)
        self.pause_step = 0.1
        self.init_position = (0.18, 0, -0.06)
        self.pause = 0
        self.gait = {"tripod_gait": {"rf": 0, "rm": pi, "rr": 0, "lf": pi, "lm": 0, "lr": pi},
                     "tetrapod_gait": {"rf": 0, "rm": pi * 2 / 3, "rr": -pi * 2 / 3, "lf": pi * 2 / 3, "lm": -pi * 2 / 3,"lr": 0},
                     "hexapod_gait": {"rf": 0, "rm": pi / 3, "rr": pi * 2 / 3, "lf": pi, "lm": -2 * pi / 3,"lr": -pi / 3},
                     "turn_right":{"rf": 0, "rm": pi, "rr": 0, "lf": pi, "lm": 0, "lr": pi},
                     "turn_left":{"rf": 0, "rm": pi, "rr": 0, "lf": pi, "lm": 0, "lr": pi}
                     }
        self.force = {"rf": 2, "rm": 2, "rr": 2, "lf": 2, "lm": 2,"lr": 2}
        self.move = {"rf": 0.05, "rm": 0.05, "rr": 0.05, "lf": 0.05, "lm": 0.05,"lr": 0.05}
        self.high = {"rf": 0.05, "rm": 0.05, "rr": 0.05, "lf": 0.05, "lm": 0.05,"lr": 0.05}
        self.gait_choose = {0:"tripod_gait",1:"tetrapod_gait",2:"hexapod_gait",3:"turn_right",4:"turn_left"}
    def forward_kinematics(self,theta_list):
        """
        * @brief 正运动学求解
        * @param theta_list为列表，包括[theta1, theta2, theta3]使用弧度制
        * @return 足尖相对于固定点的x,y,z坐标使用单位m
        * @note 基于改进DH参数表使用正运动学算法实现
        * @warning 输入角度超出范围可能不会报错，但运动学范围需外部保证，建议[-5pi/6,5pi/6]之间
        """
        theta1, theta2, theta3 = theta_list
        x = cos(theta1) * (self.L2 * cos(theta2) + self.L1 + self.L3 * cos(theta2 - theta3 + pi / 4))
        y = sin(theta1) * (self.L2 * cos(theta2) + self.L1 + self.L3 * cos(theta2 - theta3 + pi / 4))
        z = -self.L2 * sin(theta2) - self.L3 * sin(theta2 - theta3 + pi / 4)
        return [x, y, z]
    def inverse_kinematics(self,position_list):
        """
        * @brief 逆运动学求解
        * @param position_list为列表，包括[x,y,z]使用单位m
        * @return 每个关节的目标电机角度使用弧度制
        * @note 基于改进DH参数表使用逆运动学算法实现
        * @warning 输入位置超出求解范围可能不会报错，但运动学范围需外部保证
        """
        x, y, z = position_list
        theta1 = atan2(y, x)
        theta2 = atan2((x * cos(theta1) + y * sin(theta1) - self.L1), z) + asin(
            ((self.L3 ** 2 - self.L2 ** 2 - z ** 2 - (x * cos(theta1) + y * sin(theta1) - self.L1) ** 2) / (2 * self.L2)) / sqrt(
                z ** 2 + (x * cos(theta1) + y * sin(theta1) - self.L1) ** 2))
        theta3 = pi / 4 + theta2 - atan2((-self.L2 * sin(theta2) - z) / self.L3,
                                         (x * cos(theta1) + y * sin(theta1) - self.L1 - self.L2 * cos(theta2)) / self.L3)
        return [theta1, theta2, theta3]
    def change_leg_posture_by_position(self,position_list, name, force):
        """
        * @brief 改变腿的姿态
        * @param x,y,z为坐标,name为字符串可以有以下取值
                 "rf"表示右前腿，"rm"表示右中腿，"rr"表示右后腿
                 "lf"表示左前腿，"lm"表示左中腿，"lr"表示左后腿
        * @return None
        * @note 基于逆运动学算法和电机角度控制实现
        * @warning 输入位置超出求解范围可能不会报错，但运动学范围需外部保证
        """
        theta1, theta2, theta3 = self.inverse_kinematics(position_list)
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][0], controlMode=p.POSITION_CONTROL,
                                targetPosition=theta1, force=force)
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][1], controlMode=p.POSITION_CONTROL,
                                targetPosition=theta2, force=force)
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][2], controlMode=p.POSITION_CONTROL,
                                targetPosition=theta3, force=force)
    def change_leg_posture_by_theta(self,theta_list, name, force):
        """
        * @brief 改变腿的姿态
        * @param theta1,theta2,theta3为目标角度,name为字符串可以有以下取值
                 "rf"表示右前腿，"rm"表示右中腿，"rr"表示右后腿
                 "lf"表示左前腿，"lm"表示左中腿，"lr"表示左后腿
        * @return None
        * @note 基于电机角度控制实现
        * @warning 角度须在-pi*5/6到pi*5/6之间
        """
        theta1, theta2, theta3 = theta_list
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][0],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=theta1, force=force)
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][1],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=theta2, force=force)
        p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=self.joint_name_to_id[name][2],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=theta3, force=force)
    def get_leg_posture(self):
        """
        * @brief 获得腿的状态，包括角度，角速度,扭矩等参数
        * @param None
        * @return 返回为列表，内容为18个角度，18个角速度，18个扭矩
        * @note None
        * @warning None
        """
        ans = [sublist[i] for i in (0, 1, 3) for sublist in p.getJointStates(self.hexapod, [num for lst in self.joint_name_to_id.values() for num in lst])]
        return ans
    def get_euler(self):
        """
        * @brief 获得模型的欧拉角参数
        * @param None
        * @return 返回为列表，内容为roll,pitch,yaw，范围限制在roll[-pi,pi],pitch[-pi/2,pi/2],yaw[-pi,pi]
        * @note None
        * @warning None
        """
        ans = list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.hexapod)[1]))
        return ans
    def get_speed(self):
        """
        * @brief 获得模型的速度参数
        * @param None
        * @return 返回为列表，内容为Vx,Vy,Vz，无范围限制
        * @note None
        * @warning None
        """
        return [i for i in p.getBaseVelocity(self.hexapod)[0]]
    def hight(self):
        """
        * @brief 获得模型的高度参数
        * @param None
        * @return 返回为列表，内容为z轴高度
        * @note None
        * @warning None
        """
        return [p.getBasePositionAndOrientation(self.hexapod)[0][2]]
    def get_observation_space(self):
        """
        * @brief 获得模型的观察空间
        * @param None
        * @return 返回为numpy数组
        * @note None
        * @warning None
        """
        return np.array(self.get_leg_posture()+self.hight()+self.get_euler()+self.get_speed()+[self.target],dtype=np.float32)
    def reset(self):
        """
        * @brief 将模型重置回原始点
        * @param None
        * @return None
        * @note 将模型坐标和欧拉角重置，关节状态与目标也全部重置
        * @warning None
        """
        p.resetBasePositionAndOrientation(self.hexapod,[0,0,0.5],p.getQuaternionFromEuler([0,0,0]))
        for name in self.joint_name_to_id:
            for joint_id in self.joint_name_to_id[name]:
                p.resetJointState(self.hexapod, joint_id, targetValue=0, targetVelocity=0)
                p.setJointMotorControl2(bodyIndex=self.hexapod, jointIndex=joint_id,controlMode=p.POSITION_CONTROL,targetPosition=0, force=self.init_force)
        self.pause = 0
        self.target = random.uniform(-pi/2, pi/2)
    def change_next_theta(self,selected_gait):
        """
        * @brief 对机器人进行下一步电机角度计算与更新，并且返回一个周期是否完结的标志位
        * @param selected_gait是一个字符串，可以取值为
                                                    "tripod_gait"         三角步态
                                                    "tetrapod_gait"       四角步态
                                                    "hexapod_gait"        六角步态
                                                    "turn_right"          右转
                                                    “turn_left”           左转
        * @return 一个标志位，反应一个相位周期是否完成
        * @note None
        * @warning None
        """
        last_pause = self.pause
        self.pause = remainder(self.pause + self.pause_step, 2 * pi)
        flag = last_pause <= 0 < self.pause
        for name in self.joint_name_to_id:
            temp = remainder(self.pause + self.gait[selected_gait][name], 2 * pi)
            if name[0] == "l":
                s = 1
            else:
                s = -1
            x = self.init_position[0] - 0.02
            if selected_gait == "tripod_gait":
                if temp > 0:            #摆动相
                    y = self.init_position[1] + s * self.move[name] * cos(temp)
                    z = self.init_position[2] + self.high[name] * sin(temp)
                else:                   #支撑相
                    y = self.init_position[1] + s * self.move[name] * cos(temp)
                    z = self.init_position[2]
            elif selected_gait == "tetrapod_gait":
                if 0 <= temp < pi*2/3:  #摆动相
                    y = self.init_position[1] + s * self.move[name] * cos(temp * 1.5)
                    z = self.init_position[2] + self.high[name] * sin(temp * 1.5)
                else:                   #支撑相
                    y = self.init_position[1] + s * self.move[name] * cos(temp * 0.75)
                    z = self.init_position[2]
            elif selected_gait == "hexapod_gait":
                if 0 <= temp < pi/3:    #摆动相
                    y = self.init_position[1] + s * self.move[name] * cos(temp*3)
                    z = self.init_position[2] + self.high[name] * sin(temp*3)
                else:                   #支撑相
                    y = self.init_position[1] + s * self.move[name] * cos(temp*0.6)
                    z = self.init_position[2]
            elif selected_gait == "turn_right":
                if temp > 0:            #摆动相
                    y = self.init_position[1] + self.move[name] * cos(temp)
                    z = self.init_position[2] + self.high[name] * sin(temp)
                else:                   #支撑相
                    y = self.init_position[1] + self.move[name] * cos(temp)
                    z = self.init_position[2]
            else:
                if temp > 0:            #摆动相
                    y = self.init_position[1] - self.move[name] * cos(temp)
                    z = self.init_position[2] + self.high[name] * sin(temp)
                else:                   #支撑相
                    y = self.init_position[1] - self.move[name] * cos(temp)
                    z = self.init_position[2]
            self.change_leg_posture_by_position([x,y,z],name,self.force[name])
        return not flag
class Camera:
    """
    这是一个摄像机类用于控制视角
    有两个函数：
        1.__init__函数用于初始化
            示例：camera = Camera()
        2.camera_change函数用于监听键盘改变摄像机参数
            示例：camera.camera_change()
    """
    def __init__(self):
        """
        * @brief 摄像机位置初始化
        * @param camera_distance为距离,camera_yaw为水平角度,camera_pitch为俯仰角,camera_target_positon为摄像机看的位置坐标
        * @return None
        """
        self.camera_distance = 3
        self.camera_yaw = 0
        self.camera_pitch = -45
        self.camera_target_positon = (0,0,0.2)
        p.resetDebugVisualizerCamera(self.camera_distance, self.camera_yaw, self.camera_pitch,self.camera_target_positon)
    def camera_change(self):
        """
            * @brief 监听按键使摄像机位置改变
                     按下k向下，按下i向上，按下l向左，按下j向右
            * @param None
            * @return None
        """
        keys = p.getKeyboardEvents()
        for k,v in keys.items():
            if v & p.KEY_WAS_TRIGGERED:
                if k == ord("k"):
                    self.camera_pitch += 5
                elif k == ord("i"):
                    self.camera_pitch -= 5
                elif k == ord("l"):
                    self.camera_yaw += 5
                elif k == ord("j"):
                    self.camera_yaw -= 5
        p.resetDebugVisualizerCamera(self.camera_distance,self.camera_yaw,self.camera_pitch,self.camera_target_positon)
class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        low = np.array([-5*pi/6]*18+   #六个腿6组角度参数
                       [-5]*18+   #角速度数据
                       [-5]*18+   #力矩数据
                        [0]+            #z轴高度
                       [-pi,-pi/2,-pi]+        #姿态欧拉角
                       [-5,-5,-5]+  #x,y,z方向速度
                       [-pi])            #目标角度
        high = np.array([5*pi/6]*18+   #六个腿6组角度参数
                        [5]*18+   #角速度数据
                        [5]*18+   #力矩数据
                        [3]+            #z轴高度
                       [pi,pi/2,pi]+#姿态欧拉角
                       [5,5,5]+     #x,y,z方向速度
                       [pi])         #目标角度
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        low = np.array([0]*12+      #高度与位移距离
                       [0.01]+      #单位相位
                       [1.5]*6+     #力矩
                       [0])         #姿态选择
        high = np.array([0.06]*6+   #高度
                        [0.09]*6+   #位移
                        [0.2]+      #单位相位
                        [8]*6+      #力矩
                        [5])        #姿态选择
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        global hexapod
        hexapod.reset()
        hf.UpdateHeightField()
        self.state = hexapod.get_observation_space()
        self.steps = 0
        return self.state, {}
    def step(self, action):
        global hexapod
        #action为一个20维的数组
        (hexapod.high["rf"],hexapod.high["rm"],hexapod.high["rr"],hexapod.high["lf"],hexapod.high["lm"],hexapod.high["lr"],
        hexapod.move["rf"],hexapod.move["rm"],hexapod.move["rr"],hexapod.move["lf"],hexapod.move["lm"],hexapod.move["lr"],
        hexapod.pause_step,
        hexapod.force["rf"],hexapod.force["rm"],hexapod.force["rr"],hexapod.force["lf"],hexapod.force["lm"],hexapod.force["lr"]) = action[:19]

        action[19] = int(action[19])
        if action[19] == 5:
            action[19] = 4

        while hexapod.change_next_theta(hexapod.gait_choose[action[19]]):
            p.stepSimulation()
        p.stepSimulation()

        self.state = hexapod.get_observation_space()
        self.steps += 1

        reward = (-0.1*sum(abs(x) for x in self.state[36:54]) +                                                         #力矩惩罚
                  -0.3*((4*self.state[-7])**2+(3*abs(self.state[-6]))**2)                                               #欧拉角倾斜惩罚
                  +4*(1/(abs((cos(self.state[-1])*self.state[-4]+sin(self.state[-1])*self.state[-3])-0.5)+1)-2/3)   #速度奖励                                    #速度奖励
                  -0.2*sqrt(abs(atan2(self.state[-3], self.state[-4]) - self.state[-1]))                          #速度方向惩罚
                  -0.1                                                                                                    #步数惩罚
                  -0.3*sqrt(abs(self.state[-5]-self.state[-1]))                                                         #朝向惩罚
                  )

        terminated = False
        if abs(self.state[-7]) > pi/3 or abs(self.state[-6]) > pi/3:
            terminated = True
        truncated = self.steps >= 1000

        return self.state, reward, terminated, truncated, {}

if __name__ == '__main__':
    #p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)

    terrain_env = heightfield.Env()
    hf = heightfield.HeightField()
    hf._generate_field(terrain_env)
    #floor = p.loadURDF(pybullet_data.getDataPath() + r"/plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

    hexapod = Hexapod(L1,L2,L3)

    env = CustomEnv()
    env = Monitor(env)
    Now_Time = time.strftime("%Y-%m-%d %H_%M_%S")
    model_path = f"./{model_name[:19]}/{model_name}.zip"

    if os.path.exists(model_path):
        print(f"发现中断前的存档：{model_path}，正在加载...")
        model = SAC.load(model_path, env=env, verbose=1)
        print(f"已训练步数恢复为：{model.num_timesteps}")
    else:
        print("未发现存档，创建全新模型...")
        os.makedirs(Now_Time, exist_ok=True)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[256, 256]))
        )

    checkpoint_callback = CheckpointCallback(
        save_freq= 50000,
        save_path= f"./{Now_Time}/",
        name_prefix = Now_Time,
        save_replay_buffer=True,       # 保存经验池
    )

    print(f"开始训练,当前步数：{model.num_timesteps}")
    model.learn(total_timesteps=150000,log_interval=10,callback=checkpoint_callback,reset_num_timesteps=False)
    print("训练完成！")

    model.save(f"./{Now_Time}/{Now_Time}.zip")
    print(f"模型已保存为 ./{Now_Time}/{Now_Time}.zip")
    env.close()


