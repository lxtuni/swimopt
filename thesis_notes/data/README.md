# data/all_runs.csv 列说明

每一行是一次完整的 CMA-ES 优化运行的**最优解**，指标由该解的计分仿真直接得到。
由 `python thesis_notes/export_data.py` 从本机 `results_*` 文件夹重新生成（UTF-8 带 BOM，Excel 可直接打开）。

| 列 | 含义 |
|---|---|
| `study` | 实验：`free_phase_servo`（实验1）、`families_speed`（实验2）、`speed_power`（实验3）、`families_speed_strict`（实验4）、`drag_vs_lift_2x2_preservo`（已撤回） |
| `status` | `valid` 或 `WITHDRAWN (...)`，撤回的不要引用 |
| `run` | 结果文件夹 |
| `physics` | `drag`（仅阻力）或 `drag+lift` |
| `family` | 步态族：free / diag(trot) / lr(pace) / fb(bound) / wave(walk) / inphase(pronk) |
| `seed` | CMA-ES 随机种子 |
| `objective`, `min_speed` | `speed`（最快）或 `power`（最小功率，要求速度 ≥ min_speed，m/s） |
| `limit_heading/roll/pitch` | RMS 限值（°），180 表示无限值 |
| `limit_surfacing` | 肢体出水时间比例上限，空 = 不约束 |
| `servo_dps` | 舵机空载转速 °/s，空 = 不限速 |
| `budget` | 评估次数 |
| `feasible` | 是否满足所有约束 |
| `speed_m_s`, `bl_s` | 前进速度（沿初始朝向），体长/秒（体长 0.239 m） |
| `power_W` | 平均机械功率（电机输出轴，绝对值，不回收） |
| `cot_J_m` | 功率/速度（J/m），未除以重量 |
| `freq_Hz`, `duty` | 划水频率，发力冲程占比 |
| `heading_rms`, `roll_rms`, `pitch_rms` | 航向偏差、横滚、俯仰的 RMS（°） |
| `impulse_lift_Ns`, `impulse_drag_Ns` | 计分窗口内升力项、阻力项沿前进方向的有符号冲量（N·s）；升力为正、阻力为负 = 升力推进 |
| `peak_joint_speed_dps` | 关节峰值转速（°/s） |
| `torque_sat` | 任一舵机处于力矩上限的时间比例 |
| `surfacing` | 任一肢体浸没比例 < 0.5 的时间比例 |
| `nearest_family`, `deviation` | 最近的步态族及平均相位偏离（周期比例，0 = 完全一致） |

注意：`results_cmp`（已撤回）的运行没有舵机模型，其 `peak_joint_speed_dps` 等有效性指标缺失（NaN）。
