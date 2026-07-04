import sys
sys.path.append("mimickit")

from argparse import ArgumentParser
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.optim as optim
import yaml

import anim.motion as motion
from learning.tinymdm.tinymdm_model import TinyMDMModel
from motion_prior_dataset import MotionPriorData
import util.logger as logger
import util.tb_logger as tb_logger

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def build_logger(log_file, logger_type):
    if logger_type == "txt":
        log = logger.Logger()
    elif logger_type == "tb":
        log = tb_logger.TBLogger()
    else:
        raise ValueError(f"Unsupported logger: {logger_type}")

    log.set_step_key("Iteration")
    log.configure_output_file(log_file)
    return log

def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"

class ProgressBar:
    def __init__(self, total, width=30, min_interval=0.5):
        self._total = total
        self._width = width
        self._min_interval = min_interval
        self._start_time = time.time()
        self._last_update_time = 0.0
        self._last_line_len = 0
        return

    def update(self, step, loss=None, force=False):
        now = time.time()
        if (not force) and (now - self._last_update_time < self._min_interval):
            return

        self._last_update_time = now
        progress = min(float(step) / float(self._total), 1.0)
        filled = int(round(self._width * progress))
        bar = "#" * filled + "-" * (self._width - filled)

        elapsed = now - self._start_time
        if step > 0:
            eta = elapsed * (self._total - step) / float(step)
        else:
            eta = 0.0

        loss_text = "" if loss is None else f" loss={loss:.4f}"
        line = (
            f"\r[{bar}] {step}/{self._total} "
            f"{100.0 * progress:5.1f}%{loss_text} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()
        self._last_line_len = len(line)
        return

    def clear(self):
        if self._last_line_len > 0:
            sys.stderr.write("\r" + " " * self._last_line_len + "\r")
            sys.stderr.flush()
        return

    def close(self):
        self.update(self._total, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()
        return

@torch.no_grad()
def generate(model, dataset, obs_space, config, device, out_motion_dir, enable_ema=False, num_samples=16):
    fps = dataset.control_freq
    num_frames = obs_space.shape[-1] // config["input_channel"]

    if enable_ema:
        gen_samples = model.sample_ema(shape=obs_space.shape, batch_size=num_samples, device=device)
    else:
        gen_samples = model.sample(shape=obs_space.shape, batch_size=num_samples, device=device)

    gen_samples = model.unnormalize(gen_samples.reshape([num_samples, num_frames, -1]))

    for i, gen_sample in enumerate(gen_samples):
        frames = dataset.convert_sample_to_frames(gen_sample).detach().cpu().numpy()
        sample_motion = motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=fps, frames=frames)

        out_motion_file = os.path.join(out_motion_dir, f"motion_{i:03}.pkl")
        sample_motion.save(out_motion_file)

        pos_sample = dataset.calc_joint_position_from_frame(frames)[None,...]
        dataset.plot_jnt(jnt_pos=pos_sample, out_path=os.path.join(out_motion_dir, f"anim_{i:03}"))
    
    return
    
@torch.no_grad()
def test(cfg_path, model_file, out_dir=None, num_samples=16, device="cuda"):
    fixseed(0)
    assert(out_dir is not None and out_dir != ""), "Must specify --out_dir"
    assert(model_file != ""), "Must specify --model_file"

    with open(cfg_path, "r") as stream:
        config = yaml.safe_load(stream)

    with open(config["env_config"], "r") as stream:
        env_config = yaml.safe_load(stream)
    
    out_motion_dir = os.path.join(out_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    dataset_env = MotionPriorData(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)

    priormodel = TinyMDMModel(config, device)
    prior_state_dict = torch.load(model_file, map_location=device)
    priormodel.load_state_dict(prior_state_dict)
    
    priormodel.eval()
    priormodel.to(device)
    
    generate(priormodel, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
            enable_ema=priormodel.model_ema, num_samples=num_samples)
    return

def train(cfg_path, out_dir=None, device="cuda", logger_type="tb"):
    assert out_dir is not None and out_dir != "", "Must specify --out_dir"
    
    with open(cfg_path, "r") as stream:
        config = yaml.safe_load(stream)
    
    env_file = config["env_config"]
    with open(env_file, "r") as stream:
        env_config = yaml.safe_load(stream)

    out_motion_dir = os.path.join(out_dir, "samples")
    out_model_file = os.path.join(out_dir, "model.pt")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_motion_dir, exist_ok=True)

    out_env_config_file = os.path.join(out_dir, "env_config.yaml")
    shutil.copy(env_file, out_env_config_file)
    
    config["env_config"] = out_env_config_file
    out_config_file = os.path.join(out_dir, "diffusion_config.yaml")
    with open(out_config_file, "w") as stream:
        yaml.dump(config, stream)

    out_log_file = os.path.join(out_dir, "log.txt")
    log = build_logger(out_log_file, logger_type)
    
    batch_size = config["batch_size"]
    num_samples_stat = config.get("num_samples_stat", 10_000)
    output_iter = config.get("output_iter", 2_000)
    grad_clip_norm = config.get("grad_clip_norm", 1.0)

    dataset_env = MotionPriorData(config, device)
    obs_space = dataset_env.get_obs_space()
    num_obs_steps = env_config["num_disc_obs_steps"]
    
    config["input_dim"] = obs_space.shape[-1]
    config["input_channel"] = int(config["input_dim"] / num_obs_steps)
    print(f"Input_channel: {config['input_channel']}")

    samples = dataset_env.fetch_obs_demo(num_samples_stat)
    model = TinyMDMModel(config, device)
    model.update_normalizer(samples)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    model.to(device)
    
    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params / 1_000_000:.2f} M")

    num_iters = config['num_iterations']
    curr_iters = 0
    loss_sum = 0
    loss_count = 0
    progress_bar = ProgressBar(num_iters)

    while curr_iters < num_iters:
        samples = dataset_env.fetch_obs_demo(batch_size).clone().detach()
        samples = samples.to(device)
        samples = model.normalize(samples.reshape(batch_size, -1, config["input_channel"])).reshape(batch_size, -1)

        loss = model(samples)
        loss_sum += loss.item()
        loss_count += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        if model.model_ema is not False:
            model.ema_dmodel.update()

        progress_bar.update(curr_iters + 1, loss=loss.item())

        if (curr_iters % output_iter == 0 and curr_iters != 0) or (curr_iters == num_iters - 1):
            progress_bar.clear()
            model.eval()
            generate(model, dataset_env, obs_space, config, device, out_motion_dir=out_motion_dir,
                    enable_ema=model.model_ema, num_samples=16)
            torch.save(model.state_dict(), out_model_file)
            model.train()

            log.log("Iteration", curr_iters, collection="0_Main")
            log.log("Loss", loss_sum / max(loss_count, 1), collection="0_Main")
            log.print_log()
            log.write_log()

            loss_sum = 0
            loss_count = 0
            progress_bar.update(curr_iters + 1, loss=loss.item(), force=True)

        curr_iters += 1

    progress_bar.close()
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--cfg_path", type=str, default="tools/diffusion_model/config/tinymdm.yaml")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logger", type=str, default="tb", choices=["txt", "tb"])
    args = parser.parse_args()

    if args.mode == "train":
        print("Training new model...")
        train(args.cfg_path, out_dir=args.out_dir, device=args.device, logger_type=args.logger)
    else:
        print("Testing model...")
        test(args.cfg_path, args.model_file, out_dir=args.out_dir, device=args.device)
