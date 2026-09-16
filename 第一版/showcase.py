import pybullet as p
import time
from stable_baselines3 import SAC
import pybullet_data
import robot
import heightfield

url = "2026-09-10 07_24_00_150000_steps"

L1 = 0.054
L2 = 0.0661
L3 = 0.08
my_force = 2

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
                if k == ord("k") or k == ord('K'):
                    self.camera_pitch += 5
                elif k == ord("i") or k == ord("I"):
                    self.camera_pitch -= 5
                elif k == ord("l") or k == ord("L"):
                    self.camera_yaw += 5
                elif k == ord("j") or k == ord("J"):
                    self.camera_yaw -= 5
                elif k == ord("u") or k == ord("U"):
                    self.camera_distance += 1
                elif k == ord("o") or k == ord("O"):
                    self.camera_distance -= 1
        p.resetDebugVisualizerCamera(self.camera_distance,self.camera_yaw,self.camera_pitch,self.camera_target_positon)

p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

#floor = p.loadURDF(pybullet_data.getDataPath() + r"/plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
terrain_env = heightfield.Env()
hf = heightfield.HeightField()
hf._generate_field(terrain_env)




hexapod = robot.Hexapod(L1,L2,L3)
hexapod.target = 0
model = SAC.load(f"./{url[:19]}/{url}.zip")

while True:
    action, _ = model.predict(hexapod.get_observation_space(), deterministic=True)
    (hexapod.high["rf"], hexapod.high["rm"], hexapod.high["rr"], hexapod.high["lf"], hexapod.high["lm"],
     hexapod.high["lr"],
     hexapod.move["rf"], hexapod.move["rm"], hexapod.move["rr"], hexapod.move["lf"], hexapod.move["lm"],
     hexapod.move["lr"],
     hexapod.pause_step,
     hexapod.force["rf"], hexapod.force["rm"], hexapod.force["rr"], hexapod.force["lf"], hexapod.force["lm"],
     hexapod.force["lr"]) = action[:19]
    action[19] = int(action[19])
    if action[19] == 5:
        action[19] = 4
    while hexapod.change_next_theta(hexapod.gait_choose[int(action[19])]):
        p.stepSimulation()
        time.sleep(1/240)



























