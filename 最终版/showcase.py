import pybullet as p
import time
from stable_baselines3 import SAC
import robot
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import heightfield
import pybullet_data
import stair

url = "2026-09-15 18_17_09_850000_steps"

L1 = 0.054
L2 = 0.0661
L3 = 0.08
my_force = 2
terrain = 2

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

if terrain == 0:
    floor = p.loadURDF(pybullet_data.getDataPath() + r"/plane.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
elif terrain == 1:
    terrain_env = heightfield.Env()
    hf = heightfield.HeightField()
    hf._generate_field(terrain_env)
else:
    st = stair.StairTerrain(step_height=0.08, step_depth=0.4)
hexapod = robot.Hexapod(L1,L2,L3)
hexapod.target = 0

model = SAC.load(f"./{url[:19]}/{url}.zip")
env = DummyVecEnv([robot.make_env])
env = VecNormalize.load(f"./{url[:19]}/{url[:19]}_vecnormalize_{model.num_timesteps}_steps.pkl", env)
env.training = False       # 推理模式，不再更新归一化统计

while True:
    action, _ = model.predict(env.normalize_obs(hexapod.get_observation_space()), deterministic=True)
    hexapod.update(action)
    #time.sleep(1/240)




























