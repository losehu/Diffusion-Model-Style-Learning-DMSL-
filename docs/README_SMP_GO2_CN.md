# SMP-Go2 Trot Track 中文说明

当前仓库只保留 Go2 的 `trot_track` 版本。这个版本使用 APEX 的 Go2 trot 数据，训练 TinyMDM motion prior，再用 SMP policy 跟踪该步态。

## 保留的配置

```text
data/envs/smp_go2_trot_track_env.yaml
data/agents/smp_go2_trot_track_agent.yaml
tools/diffusion_model/config/tinymdm_go2_trot_track.yaml
args/smp_go2_trot_track_args.txt
```

当前满意的训练结果保留在：

```text
output/smp_go2_trot_track_reward/
```

其中 policy 是：

```text
output/smp_go2_trot_track_reward/model.pt
```

policy 需要的 prior 已经放进同一个结果目录：

```text
output/smp_go2_trot_track_reward/prior/
```

## 训练 Prior

每次启动都会用 `月日时分秒` 创建新目录，例如 `output/smp_prior_go2_trot_track_0618123059`。

```bash
scripts/train_go2_trot_track_prior.sh
```

指定设备：

```bash
DEVICE=cuda:0 scripts/train_go2_trot_track_prior.sh
```

## 训练 Policy

每次启动都会用 `月日时分秒` 创建新目录，例如 `output/smp_go2_trot_track_0618123059`。

```bash
scripts/train_go2_trot_track_policy.sh
```

如果刚训练过 timestamp prior，policy 脚本会自动使用最新的 `output/smp_prior_go2_trot_track_*`。也可以手动指定：

```bash
PRIOR_DIR=output/smp_prior_go2_trot_track_0618123059 \
scripts/train_go2_trot_track_policy.sh
```

双卡训练：

```bash
scripts/train_go2_trot_track_policy.sh --devices cuda:0 cuda:1
```

调整训练量：

```bash
MAX_SAMPLES=2621440000 scripts/train_go2_trot_track_policy.sh
```

训练日志重点看：

```text
Train_Return
Test_Return
Task_Reward_Mean
Combined_Reward_Mean
Root_Pos_Err
Root_Vel_Err
Test_Episode_Length
```

`Task_Reward_Mean` 不应长期接近 0；`Test_Episode_Length` 越接近 300，说明越不容易提前终止。

## 测试和可视化

默认测试当前保留的满意结果，端口是 6006：

```bash
scripts/test_go2_trot_track_policy.sh
```

测试指定模型：

```bash
scripts/test_go2_trot_track_policy.sh output/smp_go2_trot_track_0618123059/model.pt
```

换端口：

```bash
PORT=6008 scripts/test_go2_trot_track_policy.sh
```

浏览器打开：

```text
http://服务器IP:6006
```

## 关键说明

`smp_go2_trot_track_env.yaml` 里打开了：

```yaml
enable_deepmimic_reward: True
enable_tar_obs: True
pose_termination: True
```

这几个开关是 Go2 能持续跟踪 trot 的关键。旧配置里 tracking reward 没真正接上，容易出现机器狗先走几步然后停住的问题。
