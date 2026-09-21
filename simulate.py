# -*- coding: utf-8 -*-
"""
仿真评估 — 把 ①模型 ②水动力 ③步态 串起来, 跑一次给一个分数.
优化器只通过 evaluate(x) 这一个接口与仿真交互, 完全不需要知道机器人细节.
"""
import json, numpy as np, mujoco
from hydro import HydroModel
from gait import SineGait


class Swimmer:
    def __init__(self, xml_path, cfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        if "timestep" in cfg:
            self.model.opt.timestep = float(cfg["timestep"])
        self.hydro = HydroModel(self.model, cfg.get("hydro", {}))
        self.gait = SineGait(self.model, cfg.get("gait", {}))
        self.data = mujoco.MjData(self.model)
        self.T = float(cfg.get("sim_time", 12.0))
        self.settle = float(cfg.get("settle_time", 1.0))   # 起始静置(等浮态稳定)
        self.vmax_guard = float(cfg.get("vmax_guard", 5.0))
        self.trunk = cfg.get("trunk_body", None)
        self.trunk_id = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.trunk)
                         if self.trunk else 1)
        self.w_yaw = float(cfg.get("w_yaw", 0.3))          # 偏航惩罚权重
        self.body_len = float(cfg.get("body_length", 0.0)) # 体长(m), 用于报告 BL/s; 0=自动估
        if self.body_len <= 0:
            # 从模型几何自动估: 所有geom包围盒的x向跨度
            import numpy as _np
            lo, hi = 1e9, -1e9
            for g in range(self.model.ngeom):
                x = float(self.model.geom_pos[g][0]); r = float(self.model.geom_rbound[g])
                lo = min(lo, x-r); hi = max(hi, x+r)
            self.body_len = max(hi-lo, 1e-3)
        self.w_energy = float(cfg.get("w_energy", 0.0))    # 能耗惩罚(0=只看速度)

    # ---------- 单次 rollout ----------
    def rollout(self, x, record=False):
        x = self.gait.expand(x)          # 支持"只优化部分参数"的缩减向量
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)                # 确定性复位(MuJoCo 优势: 瞬时+完全一致)
        p = self.gait.decode(x)
        dt = m.opt.timestep
        n_settle = int(self.settle/dt)
        n_steps = int(self.T/dt)
        traj = []
        # 静置阶段: 不驱动, 等浮态稳定
        d.ctrl[:] = 0
        for _ in range(n_settle):
            self.hydro.apply(d); mujoco.mj_step(m, d)
        p0 = d.xpos[self.trunk_id].copy()
        yaw0 = self._yaw()
        energy = 0.0
        blew_up = False
        pitch_max = roll_max = 0.0
        for k in range(n_steps):
            t = k*dt
            d.ctrl[:] = self.gait.ctrl(p, t)
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qvel)) or np.max(np.abs(d.qvel)) > self.vmax_guard*20:
                blew_up = True; break
            v = np.linalg.norm(d.cvel[self.trunk_id, 3:6])
            if v > self.vmax_guard:
                blew_up = True; break
            energy += float(np.sum(np.abs(d.actuator_force*d.actuator_velocity)))*dt
            R = d.xmat[self.trunk_id].reshape(3, 3)
            pitch_max = max(pitch_max, abs(float(np.arcsin(np.clip(-R[2, 0], -1, 1)))))
            roll_max = max(roll_max, abs(float(np.arctan2(R[2, 1], R[2, 2]))))
            if record and k % 50 == 0:
                traj.append([t, *d.xpos[self.trunk_id], np.degrees(self._yaw()-yaw0)])
        if blew_up:
            return dict(ok=False, fitness=-1e3, dist=0.0, speed=0.0, bl_s=0.0, yaw=0.0,
                        pitch_amp=0.0, roll_amp=0.0, energy=0.0, power=0.0, traj=traj)
        p1 = d.xpos[self.trunk_id].copy()
        disp = p1 - p0
        yaw_drift = abs(np.degrees(self._wrap(self._yaw()-yaw0)))
        # 沿初始朝向投影的净位移(自动惩罚打转), 水平面内
        fwd = self._forward_dir(yaw0)
        dist = float(disp[:2] @ fwd)
        speed = dist/self.T
        power = energy/self.T                      # 平均功率(W), 与仿真时长无关
        fit = speed - self.w_yaw*np.radians(yaw_drift)/self.T - self.w_energy*power
        return dict(ok=True, fitness=float(fit), dist=dist, speed=speed,
                    bl_s=float(speed/self.body_len),
                    yaw=float(yaw_drift), pitch_amp=float(np.degrees(pitch_max)),
                    roll_amp=float(np.degrees(roll_max)),
                    energy=float(energy), power=float(power), traj=traj)

    def evaluate(self, x):
        """CMA-ES 用: 最小化 → 返回 -fitness"""
        return -self.rollout(x)["fitness"]

    # ---------- 小工具 ----------
    def _yaw(self):
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])

    @staticmethod
    def _wrap(a):
        return (a+np.pi) % (2*np.pi) - np.pi

    @staticmethod
    def _forward_dir(yaw0):
        return np.array([np.cos(yaw0), np.sin(yaw0)])

    def info(self):
        return "\n".join([self.hydro.summary(), self.gait.info(), self.gait.opt_summary(),
                          f"[sim] dt={self.model.opt.timestep*1000:.1f}ms 时长={self.T}s "
                          f"目标: fitness = 速度 - {self.w_yaw}×偏航率 - {self.w_energy}×平均功率"])


def strip_jsonc(text):
    """去掉 // 行注释和 /* */ 块注释(字符串内的不动), 让 config 可以写中文说明."""
    out, i, n = [], 0, len(text)
    in_str = in_line = in_blk = False
    esc = False
    while i < n:
        c = text[i]; nxt = text[i+1] if i+1 < n else ""
        if in_line:
            if c == "\n": in_line = False; out.append(c)
        elif in_blk:
            if c == "*" and nxt == "/": in_blk = False; i += 1
        elif in_str:
            out.append(c)
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        else:
            if c == "/" and nxt == "/": in_line = True; i += 1
            elif c == "/" and nxt == "*": in_blk = True; i += 1
            else:
                if c == '"': in_str = True
                out.append(c)
        i += 1
    return "".join(out)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(strip_jsonc(f.read()))
