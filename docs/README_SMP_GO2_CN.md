# SMP-Go2 中文说明

这份文档记录当前仓库里 Go2 版 SMP 的实际训练和测试流程。现在主要有两条路线：

- `steering`：原版 SMP 风格的速度控制，可以在网页里调 `cmd vx/cmd vy`。
- `track`：DeepMimic tracking 风格，稳定复现单个专家动作，但速度基本跟专家数据走。

当前推荐优先使用 `steering` 做速度控制；`track` 适合作为动作复现和稳定性基线。

## 目录和数据

Go2 使用 APEX 数据转换后的动作：

```text
data/motions/go2/go2_apex_trot.pkl
data/motions/go2/go2_apex_pace.pkl
data/motions/go2/go2_apex_canter.pkl
data/motions/go2/go2_apex_jump.pkl
data/motions/go2/go2_apex_jump_clean.pkl
```

每个 gait 都要先训练自己的 SMP prior，再训练 policy。不要拿 `trot` prior 去训 `jump` policy。

`go2_apex_jump.pkl` 的末尾有一帧回到起点的根位置跳变，会造成极大的假速度。训练 jump 时优先使用 `go2_apex_jump_clean.pkl`。

如果 clean 文件不存在，用下面的命令重新生成：

```bash
python tools/gmr_to_mimickit/apex_go2_csv_to_mimickit.py \
  --input_file data/motions/go2/apex_csv/go2_retarget_jump.csv \
  --output_file data/motions/go2/go2_apex_jump_clean.pkl \
  --fps 50 \
  --loop_mode 0 \
  --drop_final_wrap_threshold 1.0
```

## Steering 速度控制

相关文件：

```text
data/envs/smp_go2_steering_env.yaml
data/agents/smp_go2_steering_agent.yaml
tools/diffusion_model/config/tinymdm_go2_steering.yaml
args/smp_go2_steering_args.txt
scripts/train_go2_steering_prior.sh
scripts/train_go2_steering_policy.sh
scripts/test_go2_steering_policy.sh
tools/visualize_go2_policy_viser.py
```

核心配置对齐原版 SMP steering：

```yaml
env_name: "task_steering"
enable_tar_obs: False
pose_termination: False
enable_gsi: True
task_reward_weight: 0.5
smp_reward_weight: 0.5
```

Go2 必须保留自己的机器人配置：

```yaml
char_file: "data/assets/go2/go2.xml"
key_bodies: ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
contact_bodies: ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
```

### 训练 Trot Prior

prior 默认训练 `200_000` iterations，输出目录会自动带月日时分秒：

```bash
GAIT=trot ./scripts/train_go2_steering_prior.sh
```

临时改训练轮数：

```bash
GAIT=trot PRIOR_ITERS=50000 ./scripts/train_go2_steering_prior.sh
```

指定设备：

```bash
GAIT=trot DEVICE=cuda:0 ./scripts/train_go2_steering_prior.sh
```

### 训练 Trot Policy

APEX 的 Go2 数据主要是向前 gait，不是原版人形那种多方向、多速度 locomotion 数据。所以训练可控速度时，先用前向速度分布：

```bash
GAIT=trot RAND_TAR_DIR=False SPEED_MIN=0.8 SPEED_MAX=1.6 \
REWARD_TAR_W=1.0 REWARD_FACE_W=0.0 REWARD_VEL_SCALE=4.0 \
TASK_REWARD_WEIGHT=0.8 SMP_REWARD_WEIGHT=0.2 \
./scripts/train_go2_steering_policy.sh
```

这组参数的含义：

```text
RAND_TAR_DIR=False          只训前向速度，先别训全方向
SPEED_MIN/SPEED_MAX         目标速度采样范围
REWARD_TAR_W=1.0            速度奖励权重
REWARD_FACE_W=0.0           关闭朝向奖励，避免站着也拿高分
REWARD_VEL_SCALE=4.0        加重速度误差惩罚
TASK_REWARD_WEIGHT=0.8      更重视速度任务
SMP_REWARD_WEIGHT=0.2       prior 仍保留动作风格约束
```

如果不传这些环境变量，脚本会使用更接近原版 SMP 的默认值：

```text
RAND_TAR_DIR=True
SPEED_MIN=0.5
SPEED_MAX=5.0
REWARD_TAR_W=0.7
REWARD_FACE_W=0.3
REWARD_VEL_SCALE=0.5
TASK_REWARD_WEIGHT=0.5
SMP_REWARD_WEIGHT=0.5
```

这套默认值对原版人形 locomotion 数据合适，但对当前 APEX Go2 单 gait 容易学成站桩。

### 训练其他步态

每个 gait 都先训 prior，再训 policy。

Pace：

```bash
GAIT=pace ./scripts/train_go2_steering_prior.sh

GAIT=pace RAND_TAR_DIR=False SPEED_MIN=0.5 SPEED_MAX=1.3 \
REWARD_TAR_W=1.0 REWARD_FACE_W=0.0 REWARD_VEL_SCALE=4.0 \
TASK_REWARD_WEIGHT=0.8 SMP_REWARD_WEIGHT=0.2 \
./scripts/train_go2_steering_policy.sh
```

Canter：

```bash
GAIT=canter ./scripts/train_go2_steering_prior.sh

GAIT=canter RAND_TAR_DIR=False SPEED_MIN=1.0 SPEED_MAX=2.4 \
REWARD_TAR_W=1.0 REWARD_FACE_W=0.0 REWARD_VEL_SCALE=4.0 \
TASK_REWARD_WEIGHT=0.8 SMP_REWARD_WEIGHT=0.2 \
./scripts/train_go2_steering_policy.sh
```

Jump：

```bash
GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl \
./scripts/train_go2_steering_prior.sh

GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl \
RAND_TAR_DIR=False SPEED_MIN=0.0 SPEED_MAX=0.8 \
REWARD_TAR_W=1.0 REWARD_FACE_W=0.0 REWARD_VEL_SCALE=4.0 \
TASK_REWARD_WEIGHT=0.8 SMP_REWARD_WEIGHT=0.2 \
./scripts/train_go2_steering_policy.sh
```

`jump` 不是连续速度 gait，更适合 tracking/imitation。用 steering 训练时不要期待它像 trot/pace/canter 一样稳定调速。如果出现趴地鬼畜前进，通常是速度奖励被钻空子，优先改用 track 路线或缩小 `SPEED_MAX`。

policy 脚本会检查 prior 的 `motion_file` 是否和当前 `MOTION_FILE` 一致。如果之前用坏的 `go2_apex_jump.pkl` 训练过 jump prior，不要复用那个 prior，必须用 clean 文件重新训练 prior。

### 测试 Steering Policy

测试最新的某个 gait：

```bash
GAIT=trot ./scripts/test_go2_steering_policy.sh
```

指定权重路径：

```bash
./scripts/test_go2_steering_policy.sh output/smp_go2_trot_steering_0625232737/model.pt
```

手动指定三件套：

```bash
MODEL_FILE=output/smp_go2_trot_steering_0625232737/model.pt \
ENV_CONFIG=output/smp_go2_trot_steering_0625232737/env_config.yaml \
AGENT_CONFIG=output/smp_go2_trot_steering_0625232737/agent_config.yaml \
./scripts/test_go2_steering_policy.sh
```

换端口：

```bash
PORT=6008 ./scripts/test_go2_steering_policy.sh output/smp_go2_trot_steering_0625232737/model.pt
```

打开网页：

```text
http://服务器IP:6006
```

网页控制项：

```text
cmd vx      前向速度
cmd vy      侧向速度
Action noise 动作噪声，正常测试先保持 0
```

`tools/visualize_go2_policy_viser.py` 会直接更新 `task_steering` 的 `_tar_speed/_tar_dir` 并重新计算观测，速度命令在 policy obs 的最后 5 维。

## Track 动作复现

如果目标是稳定复现 APEX trot 专家动作，而不是调速，用 track 路线：

```text
data/envs/smp_go2_trot_track_env.yaml
data/agents/smp_go2_trot_track_agent.yaml
tools/diffusion_model/config/tinymdm_go2_trot_track.yaml
args/smp_go2_trot_track_args.txt
scripts/train_go2_trot_track_prior.sh
scripts/train_go2_trot_track_policy.sh
scripts/test_go2_trot_track_policy.sh
```

track 的关键开关：

```yaml
env_name: "smp"
enable_deepmimic_reward: True
enable_tar_obs: True
pose_termination: True
```

训练 prior：

```bash
GAIT=trot ./scripts/train_go2_trot_track_prior.sh
```

训练 policy：

```bash
GAIT=trot ./scripts/train_go2_trot_track_policy.sh
```

测试：

```bash
GAIT=trot ./scripts/test_go2_trot_track_policy.sh
```

track policy 适合看动作是否能稳定跟踪。它没有真正的速度命令输入，所以不要用它判断速度控制效果。

训练 jump track 时也建议使用 clean 数据：

```bash
GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl \
./scripts/train_go2_trot_track_prior.sh

GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl \
./scripts/train_go2_trot_track_policy.sh
```

## 日志怎么看

policy 训练重点看：

```text
Train_Return
Test_Return
Train_Episode_Length
Test_Episode_Length
Task_Reward_Mean
Smp_Reward_Mean
Combined_Reward_Mean
Sds_Loss_Mean
```

对 steering 来说：

```text
Task_Reward_Mean 高，只说明速度奖励高，不一定视觉上好看。
Smp_Reward_Mean 高，说明更像 prior 动作，但太高也可能压住速度任务。
Test_Episode_Length 接近 600，说明 20 秒 episode 没提前失败。
```

如果出现“每 600 step reset 后走两下，然后原地不动”，这是典型站桩局部最优。原因通常是原版 steering reward 对低速太松，原地不动也有不低奖励。优先使用上面的严格速度奖励命令重新训练。

## TensorBoard

policy 脚本默认可用普通 `log.txt`。如果要 TensorBoard，可以在运行参数里加：

```bash
GAIT=trot ./scripts/train_go2_steering_policy.sh --logger tb
```

查看：

```bash
tensorboard --logdir=output --port=6006 --samples_per_plugin scalars=999999
```

如果 6006 已经被可视化占用，换一个端口：

```bash
tensorboard --logdir=output --port=6008 --samples_per_plugin scalars=999999
```

## 常见问题

### No steering prior found

报错：

```text
No steering prior found for GAIT=jump.
Run: GAIT=jump scripts/train_go2_steering_prior.sh
```

说明这个 gait 的 prior 还没训练。先跑：

```bash
GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl ./scripts/train_go2_steering_prior.sh
```

然后再跑 policy。

### RAND_TAR_DIR 写法

环境变量必须写在同一条命令里：

```bash
GAIT=jump MOTION_FILE=data/motions/go2/go2_apex_jump_clean.pkl RAND_TAR_DIR=False SPEED_MIN=0.0 SPEED_MAX=0.8 ./scripts/train_go2_steering_policy.sh
```

不要写成：

```bash
RAND_TAR_DIR=
False
```

这会被 shell 拆成两条命令。

### 训练到多少停止

默认：

```text
MAX_SAMPLES=1310720000
```

可以临时加大：

```bash
MAX_SAMPLES=2621440000 GAIT=trot ./scripts/train_go2_steering_policy.sh
```

也可以先短跑看趋势：

```bash
MAX_SAMPLES=262144000 GAIT=trot ./scripts/train_go2_steering_policy.sh
```

### 双卡训练

policy 支持传给 `mimickit/run.py`：

```bash
GAIT=trot ./scripts/train_go2_steering_policy.sh --devices cuda:0 cuda:1
```

prior 训练是否多卡取决于 `tools/diffusion_model/train_tinymdm.py` 当前实现和启动参数；单卡更稳。

### Viser 依赖

如果可视化报：

```text
ModuleNotFoundError: No module named 'viser'
```

在当前环境安装：

```bash
pip install viser yourdfpy trimesh
```

如果 IsaacGym 报 `Ninja is required`，确认 wmp 环境的 bin 在 PATH：

```bash
export PATH=/root/micromamba/envs/wmp/bin:$PATH
```

## 当前结论

- Go2 steering 的网络输入检查过，policy obs 是 138 维，速度命令在最后 5 维，checkpoint 的 actor/critic 输入维度匹配。
- viewer 的速度命令已改成直接更新 `task_steering` 目标并重算 obs。
- 当前 Go2 速度控制能跑起来的关键不是改网络，而是用适合 APEX Go2 数据的前向速度范围和更严格的速度奖励。
- trot/pace/canter 更适合速度控制；jump 更适合 track imitation。
