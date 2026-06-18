import argparse
import os
import sys
import time
from collections import deque

import numpy as np

# Isaac Gym must be imported before torch.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

try:
    import viser
    from viser.extras import ViserUrdf
    import yourdfpy
except ImportError as e:
    raise ImportError(
        "This viewer needs viser and yourdfpy. Install them in the same env, e.g.:\n"
        "  pip install viser yourdfpy trimesh\n"
        "Original import error: {}".format(e)
    )

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util


FOOT_COLORS = np.array(
    [
        [250, 120, 120],
        [120, 250, 160],
        [120, 140, 255],
        [255, 230, 120],
    ],
    dtype=np.uint8,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_scene(server):
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(True, True)
    server.scene.add_grid(
        name="/grid_floor",
        width=80.0,
        height=80.0,
        width_segments=80,
        height_segments=80,
        plane="xy",
        cell_size=1.0,
        section_size=1.0,
        shadow_opacity=0.15,
        position=(0.0, 0.0, 0.0),
        visible=True,
    )
    return


def build_env_agent(args):
    env_config = load_yaml(args.env_config)
    if env_config.get("char_file", "") != "":
        args.char_file = env_config["char_file"]

    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, args.device, args.master_port)

    env = env_builder.build_env(
        args.env_config,
        args.engine_config,
        num_envs=1,
        device=args.device,
        visualize=False,
        record_video=False,
    )
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    return env, agent


def maybe_set_commands(env, obs, cmd_enable, cmd_x, cmd_y, cmd_yaw):
    if not (cmd_enable.value and hasattr(env, "commands")):
        return obs

    env.commands[:, 0] = cmd_x.value
    env.commands[:, 1] = cmd_y.value
    env.commands[:, 2] = cmd_yaw.value

    if obs is None or obs.ndim != 2:
        return obs

    # Best-effort patch for command-conditioned envs. The current Go2 track env does not use it.
    try:
        if obs.shape[1] >= 9:
            obs[:, 6:9] = torch.tensor(
                [cmd_x.value, cmd_y.value, cmd_yaw.value],
                device=obs.device,
                dtype=obs.dtype,
            )
    except Exception:
        pass
    return obs


def add_point_cloud(server, name, points, color, point_size):
    server.scene.add_point_cloud(
        name,
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(color, dtype=np.uint8),
        point_size=point_size,
    )
    return


def get_foot_body_ids(env, char_id):
    names = getattr(env._kin_char_model, "_body_names", [])
    body_ids = []
    for foot_name in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
        if foot_name in names:
            body_ids.append(names.index(foot_name))
    return body_ids


def destroy_env(env):
    engine = getattr(env, "_engine", None)
    if engine is None:
        return
    try:
        if hasattr(engine, "_viewer") and engine._viewer is not None:
            engine._gym.destroy_viewer(engine._viewer)
    except Exception:
        pass
    try:
        if hasattr(engine, "_sim") and engine._sim is not None:
            engine._gym.destroy_sim(engine._sim)
    except Exception:
        pass
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", default="output/smp_go2_trot_track_reward/model.pt")
    parser.add_argument("--env_config", default="data/envs/smp_go2_trot_track_env.yaml")
    parser.add_argument("--agent_config", default="output/smp_go2_trot_track_reward/agent_config.yaml")
    parser.add_argument("--engine_config", default="data/engines/isaac_gym_engine.yaml")
    parser.add_argument("--urdf_file", default="apex/resources/robots/go2/urdf/go2.urdf")
    parser.add_argument("--char_file", default="data/assets/go2/go2.xml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--master_port", type=int, default=6133)
    parser.add_argument("--trail_len", type=int, default=240)
    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        raise FileNotFoundError("Missing model file: {}".format(args.model_file))
    if not os.path.isfile(args.urdf_file):
        raise FileNotFoundError("Missing Go2 URDF file: {}".format(args.urdf_file))

    print("Building MimicKit env and loading policy...", flush=True)
    env, agent = build_env_agent(args)
    obs, info = env.reset()
    char_id = env._get_char_id()
    timestep = float(env._engine.get_timestep())

    print("Starting Viser server on {}:{}...".format(args.host, args.port), flush=True)
    server = viser.ViserServer(host=args.host, port=args.port)
    setup_scene(server)

    urdf = yourdfpy.URDF.load(
        args.urdf_file,
        load_meshes=True,
        build_scene_graph=True,
        load_collision_meshes=False,
        build_collision_scene_graph=False,
    )
    root = server.scene.add_frame("/robot", axes_length=0.0, axes_radius=0.0)
    robot = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )

    connected = {}

    @server.on_client_connect
    def _(client):
        connected[client.client_id] = client
        client.camera.position = (-2.2, -2.8, 1.4)
        client.camera.look_at = (0.0, 0.0, 0.35)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @server.on_client_disconnect
    def _(client):
        connected.pop(client.client_id, None)

    with server.gui.add_folder("Policy"):
        run_toggle = server.gui.add_checkbox("Run policy", initial_value=True)
        reset_btn = server.gui.add_button("Reset env")
        action_noise = server.gui.add_slider("Action noise", min=0.0, max=1.0, step=0.01, initial_value=0.0)
        sim_speed = server.gui.add_slider("Sim speed", min=0.1, max=2.0, step=0.05, initial_value=1.0)

    has_commands = hasattr(env, "commands")
    with server.gui.add_folder("Commands"):
        cmd_enable = server.gui.add_checkbox("Override commands", initial_value=has_commands)
        cmd_x = server.gui.add_slider("cmd vx", min=-2.0, max=2.0, step=0.05, initial_value=0.8)
        cmd_y = server.gui.add_slider("cmd vy", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        cmd_yaw = server.gui.add_slider("cmd yaw", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        cmd_note = server.gui.add_text(
            "Status",
            "env.commands found" if has_commands else "Go2 track env has no command input",
            disabled=True,
        )

    with server.gui.add_folder("Viz"):
        follow_cam = server.gui.add_checkbox("Follow camera", initial_value=True)
        show_base_trail = server.gui.add_checkbox("Base trail", initial_value=True)
        show_foot_trails = server.gui.add_checkbox("Foot trails", initial_value=True)

    with server.gui.add_folder("Stats"):
        step_text = server.gui.add_text("Step", "0", disabled=True)
        pos_text = server.gui.add_text("Base pos", "-", disabled=True)
        vel_text = server.gui.add_text("Base vel", "-", disabled=True)
        action_text = server.gui.add_text("Action norm", "-", disabled=True)
        reward_text = server.gui.add_text("Reward", "-", disabled=True)
        done_text = server.gui.add_text("Done", "0", disabled=True)

    reset_requested = {"value": False}

    @reset_btn.on_click
    def _(_event):
        reset_requested["value"] = True

    base_trail = deque(maxlen=args.trail_len)
    foot_trails = None
    foot_ids = get_foot_body_ids(env, char_id)
    if len(foot_ids) > 0:
        foot_trails = [deque(maxlen=args.trail_len) for _ in foot_ids]

    step = 0
    last_wall = time.time()
    print("Open http://{}:{}/".format("127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port), flush=True)

    try:
        while True:
            if reset_requested["value"]:
                obs, info = env.reset()
                base_trail.clear()
                if foot_trails is not None:
                    for trail in foot_trails:
                        trail.clear()
                reset_requested["value"] = False
                step = 0

            if run_toggle.value:
                obs = maybe_set_commands(env, obs, cmd_enable, cmd_x, cmd_y, cmd_yaw)
                with torch.no_grad():
                    action, _ = agent._decide_action(obs, info)
                    action_norm = float(torch.linalg.vector_norm(action[0]).detach().cpu().item())
                    if action_noise.value > 0.0:
                        action = action + action_noise.value * torch.randn_like(action)
                    obs, reward, done, info = env.step(action)

                if int(done[0].item()) != 0:
                    obs, info = env.reset(torch.tensor([0], device=args.device, dtype=torch.long))

                step += 1

            root_pos = env._engine.get_root_pos(char_id)[0].detach().cpu().numpy()
            root_rot = env._engine.get_root_rot(char_id)[0].detach().cpu().numpy()
            root_vel = env._engine.get_root_vel(char_id)[0].detach().cpu().numpy()
            dof_pos = env._engine.get_dof_pos(char_id)[0].detach().cpu().numpy()
            body_pos = env._engine.get_body_pos(char_id)[0].detach().cpu().numpy()

            robot.update_cfg(dof_pos)
            root.position = tuple(root_pos)
            root.wxyz = (float(root_rot[3]), float(root_rot[0]), float(root_rot[1]), float(root_rot[2]))

            if show_base_trail.value:
                base_trail.append(root_pos.copy())
                if len(base_trail) > 1:
                    add_point_cloud(server, "/base_trail", np.stack(base_trail), (0, 220, 200), 0.012)

            if show_foot_trails.value and foot_trails is not None:
                for i, body_id in enumerate(foot_ids):
                    foot_trails[i].append(body_pos[body_id].copy())
                    if len(foot_trails[i]) > 1:
                        add_point_cloud(server, "/foot_trail_{}".format(i), np.stack(foot_trails[i]), FOOT_COLORS[i], 0.011)

            if follow_cam.value:
                cam_pos = root_pos + np.array([-2.0, -2.4, 1.15], dtype=np.float32)
                cam_look = root_pos + np.array([0.35, 0.0, 0.25], dtype=np.float32)
                for client in connected.values():
                    try:
                        with client.atomic():
                            client.camera.position = tuple(cam_pos)
                            client.camera.look_at = tuple(cam_look)
                            client.camera.up_direction = (0.0, 0.0, 1.0)
                    except Exception:
                        pass

            step_text.value = str(step)
            pos_text.value = "{:.2f}, {:.2f}, {:.2f}".format(root_pos[0], root_pos[1], root_pos[2])
            vel_text.value = "{:.2f}, {:.2f}, {:.2f}".format(root_vel[0], root_vel[1], root_vel[2])
            if "action_norm" in locals():
                action_text.value = "{:.4f}".format(action_norm)
            if "reward" in locals():
                reward_text.value = "{:.4f}".format(float(reward[0].detach().cpu().item()))
                done_text.value = str(int(done[0].detach().cpu().item()))

            elapsed = time.time() - last_wall
            target = timestep / max(0.1, float(sim_speed.value))
            sleep_time = max(0.0, target - elapsed)
            time.sleep(sleep_time)
            last_wall = time.time()

    finally:
        destroy_env(env)


if __name__ == "__main__":
    main()
