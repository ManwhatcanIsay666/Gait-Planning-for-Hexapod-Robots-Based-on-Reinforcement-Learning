import pybullet as p
import pybullet_data
import random


class StairTerrain:
    def __init__(self, step_height=0.08, step_depth=0.5, step_width=100, num_steps=100,
                 origin=(1.0, 0.0, 0.0),
                 colors=([0.75, 0.82, 0.90, 1.0], [0.25, 0.32, 0.45, 1.0])):
        self.step_height = step_height
        self.step_depth = step_depth
        self.step_width = step_width
        self.num_steps = num_steps
        self.origin = origin
        self.colors = colors

        # 记录已创建的对象，便于 update 时清理
        self.step_ids = []
        self.plane_id = None

        self.build()

    # ==========================================================
    #  构建台阶
    # ==========================================================
    def build(self):
        # 地面只在第一次构建时加载，避免重复
        if self.plane_id is None:
            self.plane_id = p.loadURDF(
                pybullet_data.getDataPath() + r"/plane.urdf",
                [0, 0, 0],
                p.getQuaternionFromEuler([0, 0, 0]),
            )

        ox, oy, oz = self.origin

        for i in range(self.num_steps):
            top_z = oz + self.step_height * (i + 1)

            half_extents = [
                self.step_depth / 2.0,
                self.step_width / 2.0,
                (top_z - oz) / 2.0,
            ]

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

            # ---- 按台阶序号循环取色 ----
            color = self.colors[i % len(self.colors)]

            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color,
            )

            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[
                    ox + self.step_depth * (i + 0.5),
                    oy,
                    (oz + top_z) / 2.0,
                ],
            )
            p.changeDynamics(body, -1, lateralFriction=1.0, restitution=0.0)
            self.step_ids.append(body)

    # ==========================================================
    #  清理 & 更新
    # ==========================================================
    def clear(self):
        """删除当前所有台阶（保留地面）。"""
        for body in self.step_ids:
            try:
                p.removeBody(body)
            except Exception:
                pass
        self.step_ids = []

    def update(self,step_height=0.08, step_depth=0.5, step_width=100, num_steps=100):
        """
        重新生成台阶。

        参数为 None 时沿用当前值；传入新值则先更新属性再重建。
        randomize=True 时会随机制造一组参数（可与其他参数叠加使用）。
        返回 self 以支持链式调用。
        """
        self.step_height = step_height
        self.step_depth = step_depth
        self.step_width = step_width
        self.num_steps = num_steps
        self.clear()
        self.build()

    # ==========================================================
    #  便利方法
    # ==========================================================
    def top_surface_z(self) -> float:
        """最高一级台阶顶面高度。"""
        return self.origin[2] + self.step_height * self.num_steps

    def top_step_center(self):
        """最高一级台阶中心点坐标 (x, y, z)。"""
        ox, oy, oz = self.origin
        return [
            ox + self.step_depth * (self.num_steps - 0.5),
            oy,
            oz + self.step_height * (self.num_steps - 0.5),
        ]