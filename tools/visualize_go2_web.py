import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import sys

import numpy as np
import yaml

# Isaac Gym must be imported before torch. Keep this optional so the motion-only
# viewer can still run in environments without Isaac Gym.
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))

import anim.motion as motion
import anim.mjcf_char_model as mjcf_char_model


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MimicKit Go2 Web Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181b1f;
      --panel2: #20242a;
      --text: #edf0f2;
      --muted: #a9b0b8;
      --line: #333942;
      --accent: #4cc9a7;
      --warn: #e7bb61;
      --red: #f06f6f;
      --green: #75d689;
      --blue: #7ca5ff;
      --yellow: #f0d66f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
    }
    .stage {
      position: relative;
      min-width: 0;
      min-height: 0;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0) 36%),
        #0d1012;
    }
    .hud {
      position: absolute;
      left: 18px;
      top: 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      pointer-events: none;
    }
    .hud strong { color: var(--text); font-weight: 650; }
    .side {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }
    h1 {
      font-size: 18px;
      line-height: 1.25;
      margin: 0 0 14px;
      font-weight: 720;
    }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 6px;
    }
    select, button, input[type="range"] {
      width: 100%;
    }
    select, button {
      height: 34px;
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      font-weight: 650;
    }
    button:hover { border-color: var(--accent); }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      align-items: end;
    }
    .control { margin-bottom: 12px; }
    .value {
      float: right;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }
    input[type="checkbox"] {
      vertical-align: -2px;
      margin-right: 8px;
    }
    input[type="range"] {
      accent-color: var(--accent);
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .stat {
      background: var(--panel2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      min-height: 54px;
    }
    .stat span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 5px;
    }
    .stat b {
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }
    .legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      margin-right: 7px;
    }
    .error {
      color: var(--warn);
      line-height: 1.5;
      white-space: pre-wrap;
    }
    @media (max-width: 860px) {
      body { overflow: auto; }
      .app {
        height: auto;
        min-height: 100vh;
        grid-template-columns: 1fr;
      }
      .stage { height: 62vh; min-height: 420px; }
      .side { border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="stage">
      <canvas id="view"></canvas>
      <div class="hud" id="hud"></div>
    </main>
    <aside class="side">
      <h1>Go2 Motion Viewer</h1>
      <div id="error" class="error"></div>
      <div class="control">
        <label>步态</label>
        <select id="motion"></select>
      </div>
      <div class="row">
        <div class="control"><button id="play">暂停</button></div>
        <div class="control"><button id="reset">重置视角</button></div>
      </div>
      <div class="control">
        <label>帧 <span class="value" id="frameValue">0</span></label>
        <input id="frame" type="range" min="0" max="1" step="1" value="0" />
      </div>
      <div class="control">
        <label>播放速度 <span class="value" id="speedValue">1.00x</span></label>
        <input id="speed" type="range" min="0.1" max="3" step="0.05" value="1" />
      </div>
      <div class="row">
        <div class="control">
          <label>Yaw <span class="value" id="yawValue">-35</span></label>
          <input id="yaw" type="range" min="-180" max="180" step="1" value="-35" />
        </div>
        <div class="control">
          <label>Pitch <span class="value" id="pitchValue">24</span></label>
          <input id="pitch" type="range" min="-20" max="75" step="1" value="24" />
        </div>
      </div>
      <div class="row">
        <div class="control">
          <label>缩放 <span class="value" id="zoomValue">110</span></label>
          <input id="zoom" type="range" min="45" max="220" step="1" value="110" />
        </div>
        <div class="control">
          <label>轨迹长度 <span class="value" id="trailValue">180</span></label>
          <input id="trail" type="range" min="0" max="500" step="10" value="180" />
        </div>
      </div>
      <div class="control"><label><input id="follow" type="checkbox" checked />跟随根节点</label></div>
      <div class="control"><label><input id="feet" type="checkbox" checked />显示足端轨迹</label></div>
      <div class="section">
        <div class="stats">
          <div class="stat"><span>FPS</span><b id="fps">0</b></div>
          <div class="stat"><span>时长</span><b id="duration">0s</b></div>
          <div class="stat"><span>根高度</span><b id="height">0m</b></div>
          <div class="stat"><span>速度估计</span><b id="vel">0m/s</b></div>
        </div>
      </div>
      <div class="section">
        <div class="legend">
          <div><span class="dot" style="background:var(--red)"></span>FR foot</div>
          <div><span class="dot" style="background:var(--green)"></span>FL foot</div>
          <div><span class="dot" style="background:var(--blue)"></span>RR foot</div>
          <div><span class="dot" style="background:var(--yellow)"></span>RL foot</div>
        </div>
      </div>
    </aside>
  </div>
  <script>
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const el = id => document.getElementById(id);

    const state = {
      motions: [],
      data: null,
      frame: 0,
      frameFloat: 0,
      playing: true,
      speed: 1,
      lastTime: performance.now(),
      camera: { yaw: -35, pitch: 24, zoom: 110 },
      pointer: null,
    };

    const footColors = {
      "FR_foot": "#f06f6f",
      "FL_foot": "#75d689",
      "RR_foot": "#7ca5ff",
      "RL_foot": "#f0d66f",
    };

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function setError(msg) {
      el("error").textContent = msg || "";
    }

    async function loadMotions() {
      const res = await fetch("/api/motions");
      if (!res.ok) throw new Error(await res.text());
      state.motions = await res.json();
      const select = el("motion");
      select.innerHTML = "";
      for (const m of state.motions) {
        const opt = document.createElement("option");
        opt.value = m.file;
        opt.textContent = m.name;
        select.appendChild(opt);
      }
      if (state.motions.length > 0) await loadMotion(state.motions[0].file);
    }

    async function loadMotion(file) {
      setError("");
      const res = await fetch(`/api/motion?file=${encodeURIComponent(file)}`);
      if (!res.ok) throw new Error(await res.text());
      state.data = await res.json();
      state.frame = 0;
      state.frameFloat = 0;
      el("frame").max = state.data.frames.length - 1;
      el("frame").value = 0;
      el("fps").textContent = state.data.fps;
      el("duration").textContent = `${((state.data.frames.length - 1) / state.data.fps).toFixed(1)}s`;
      updateLabels();
    }

    function deg(v) { return v * Math.PI / 180; }

    function rotate(p) {
      const yaw = deg(state.camera.yaw);
      const pitch = deg(state.camera.pitch);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x = cy * p[0] - sy * p[1];
      const y0 = sy * p[0] + cy * p[1];
      const z0 = p[2];
      const y = cp * y0 - sp * z0;
      const z = sp * y0 + cp * z0;
      return [x, y, z];
    }

    function project(p, center) {
      const shifted = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
      const r = rotate(shifted);
      const scale = Number(el("zoom").value);
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      return [w * 0.5 + r[0] * scale, h * 0.58 - r[2] * scale, r[1]];
    }

    function drawGrid(center) {
      const size = 14;
      const step = 0.5;
      ctx.lineWidth = 1;
      for (let i = -size; i <= size; i++) {
        ctx.strokeStyle = i === 0 ? "rgba(180,190,200,0.35)" : "rgba(120,130,140,0.16)";
        drawLine([center[0] - size * step, center[1] + i * step, 0], [center[0] + size * step, center[1] + i * step, 0], center);
        drawLine([center[0] + i * step, center[1] - size * step, 0], [center[0] + i * step, center[1] + size * step, 0], center);
      }
    }

    function drawLine(a, b, center, color, width) {
      const pa = project(a, center);
      const pb = project(b, center);
      ctx.strokeStyle = color || "rgba(180,190,200,0.75)";
      ctx.lineWidth = width || 1;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }

    function drawPoint(p, center, color, radius) {
      const pp = project(p, center);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pp[0], pp[1], radius, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawTrail(name, bodyId, center) {
      const data = state.data;
      const trail = Number(el("trail").value);
      if (trail <= 0) return;
      const beg = Math.max(0, state.frame - trail);
      const color = footColors[name] || "#d0d5da";
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      for (let f = beg; f <= state.frame; f++) {
        const p = project(data.frames[f][bodyId], center);
        if (f === beg) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      if (!state.data) return;
      const data = state.data;
      const frame = data.frames[state.frame];
      const root = frame[0];
      const follow = el("follow").checked;
      const center = follow ? [root[0], root[1], 0.15] : [0, 0, 0.15];

      drawGrid(center);

      if (el("feet").checked) {
        for (const name of Object.keys(footColors)) {
          const id = data.body_ids[name];
          if (id !== undefined) drawTrail(name, id, center);
        }
      }

      const links = [];
      for (let i = 0; i < data.parents.length; i++) {
        const p = data.parents[i];
        if (p >= 0) links.push([p, i]);
      }
      links.sort((a, b) => project(frame[a[0]], center)[2] - project(frame[b[0]], center)[2]);
      for (const [a, b] of links) drawLine(frame[a], frame[b], center, "rgba(232,238,242,0.86)", 3);

      for (let i = 0; i < frame.length; i++) {
        const name = data.body_names[i];
        drawPoint(frame[i], center, footColors[name] || (i === 0 ? "#4cc9a7" : "#d9dee2"), i === 0 ? 5 : 3.5);
      }

      const next = Math.min(state.frame + 1, data.frames.length - 1);
      const dt = Math.max(1e-6, (next - state.frame) / data.fps);
      const r1 = data.frames[next][0];
      const speed = Math.hypot(r1[0] - root[0], r1[1] - root[1]) / dt;
      el("height").textContent = `${root[2].toFixed(3)}m`;
      el("vel").textContent = `${speed.toFixed(2)}m/s`;
      el("hud").innerHTML = `<strong>${data.name}</strong><br>frame ${state.frame}/${data.frames.length - 1}<br>root [${root.map(v => v.toFixed(2)).join(", ")}]`;
    }

    function updateLabels() {
      el("frameValue").textContent = state.frame;
      el("speedValue").textContent = `${Number(el("speed").value).toFixed(2)}x`;
      el("yawValue").textContent = el("yaw").value;
      el("pitchValue").textContent = el("pitch").value;
      el("zoomValue").textContent = el("zoom").value;
      el("trailValue").textContent = el("trail").value;
      el("play").textContent = state.playing ? "暂停" : "播放";
    }

    function tick(now) {
      const dt = (now - state.lastTime) / 1000;
      state.lastTime = now;
      if (state.data && state.playing) {
        state.frameFloat += dt * state.data.fps * state.speed;
        if (state.frameFloat >= state.data.frames.length) state.frameFloat %= state.data.frames.length;
        state.frame = Math.floor(state.frameFloat);
        el("frame").value = state.frame;
      }
      draw();
      requestAnimationFrame(tick);
    }

    el("motion").addEventListener("change", e => loadMotion(e.target.value).catch(err => setError(String(err))));
    el("play").addEventListener("click", () => { state.playing = !state.playing; updateLabels(); });
    el("reset").addEventListener("click", () => {
      state.camera = { yaw: -35, pitch: 24, zoom: 110 };
      el("yaw").value = -35; el("pitch").value = 24; el("zoom").value = 110;
      updateLabels();
    });
    el("frame").addEventListener("input", e => {
      state.frame = Number(e.target.value);
      state.frameFloat = state.frame;
      state.playing = false;
      updateLabels();
      draw();
    });
    el("speed").addEventListener("input", e => { state.speed = Number(e.target.value); updateLabels(); });
    el("yaw").addEventListener("input", e => { state.camera.yaw = Number(e.target.value); updateLabels(); draw(); });
    el("pitch").addEventListener("input", e => { state.camera.pitch = Number(e.target.value); updateLabels(); draw(); });
    el("zoom").addEventListener("input", e => { state.camera.zoom = Number(e.target.value); updateLabels(); draw(); });
    el("trail").addEventListener("input", () => { updateLabels(); draw(); });
    el("follow").addEventListener("change", draw);
    el("feet").addEventListener("change", draw);

    canvas.addEventListener("pointerdown", e => {
      canvas.setPointerCapture(e.pointerId);
      state.pointer = { x: e.clientX, y: e.clientY };
    });
    canvas.addEventListener("pointermove", e => {
      if (!state.pointer) return;
      const dx = e.clientX - state.pointer.x;
      const dy = e.clientY - state.pointer.y;
      state.pointer = { x: e.clientX, y: e.clientY };
      state.camera.yaw += dx * 0.35;
      state.camera.pitch = Math.max(-20, Math.min(75, state.camera.pitch + dy * 0.25));
      el("yaw").value = Math.round(state.camera.yaw);
      el("pitch").value = Math.round(state.camera.pitch);
      updateLabels();
    });
    canvas.addEventListener("pointerup", () => { state.pointer = null; });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const z = Number(el("zoom").value) * (e.deltaY > 0 ? 0.92 : 1.08);
      el("zoom").value = Math.max(45, Math.min(220, z));
      updateLabels();
    }, { passive: false });

    window.addEventListener("resize", () => { resize(); draw(); });
    resize();
    loadMotions().catch(err => setError(String(err)));
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def _load_motion_entries(motion_file):
    if motion_file.endswith(".yaml") or motion_file.endswith(".yml"):
        with open(motion_file, "r") as f:
            config = yaml.safe_load(f)
        return [entry["file"] for entry in config["motions"]]
    return [motion_file]


def _motion_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("go2_apex_", "").replace("go2_", "")


def _load_char_model(char_file):
    char = mjcf_char_model.MJCFCharModel("cpu")
    char.load(char_file)
    return char


def _compute_body_pos(char, motion_file):
    motion_data = motion.load_motion(motion_file)
    frames = torch.tensor(motion_data.frames, dtype=torch.float32)
    root_pos = frames[:, 0:3]
    root_exp = frames[:, 3:6]
    dof = frames[:, 6:]

    import util.torch_util as torch_util

    root_rot = torch_util.exp_map_to_quat(root_exp)
    joint_rot = char.dof_to_rot(dof)
    body_pos, _ = char.forward_kinematics(root_pos, root_rot, joint_rot)
    return {
        "name": _motion_name(motion_file),
        "file": motion_file,
        "fps": int(motion_data.fps),
        "frames": np.round(body_pos.detach().cpu().numpy(), 5).tolist(),
        "parents": char._parent_indices.astype(int).tolist(),
        "body_names": char.get_body_names(),
        "body_ids": {name: int(i) for i, name in enumerate(char.get_body_names())},
    }


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _collect_policy_rollout(args):
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import learning.base_agent as base_agent
    import util.mp_util as mp_util

    if not os.path.isfile(args.model_file):
        raise FileNotFoundError("Missing policy model: {}".format(args.model_file))

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

    obs, info = env.reset()
    char_id = env._get_char_id()
    timestep = env._engine.get_timestep()
    num_steps = max(1, int(round(args.rollout_seconds / timestep)))

    body_frames = []
    rewards = []
    done_step = None

    with torch.no_grad():
        for step in range(num_steps):
            body_pos = env._engine.get_body_pos(char_id)[0]
            body_frames.append(body_pos.detach().cpu().numpy())

            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward[0].detach().cpu().item()))

            done_val = int(done[0].detach().cpu().item())
            if done_val != 0:
                done_step = step
                if args.stop_on_done:
                    break
                obs, info = env.reset(torch.tensor([0], device=args.device, dtype=torch.long))

    body_frames = np.stack(body_frames, axis=0)

    if hasattr(env._engine, "_gym") and hasattr(env._engine, "_sim"):
        try:
            env._engine._gym.destroy_sim(env._engine._sim)
        except Exception:
            pass

    char = _load_char_model(args.char_file)
    return {
        "name": "policy_rollout",
        "file": "__policy_rollout__",
        "fps": int(round(1.0 / timestep)),
        "frames": np.round(body_frames, 5).tolist(),
        "parents": char._parent_indices.astype(int).tolist(),
        "body_names": char.get_body_names(),
        "body_ids": {name: int(i) for i, name in enumerate(char.get_body_names())},
        "rollout": {
            "model_file": args.model_file,
            "num_frames": int(body_frames.shape[0]),
            "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
            "sum_reward": float(np.sum(rewards)) if len(rewards) > 0 else 0.0,
            "done_step": done_step,
        },
    }


class ViewerServer:
    def __init__(self, char_file, motion_file, rollout_data=None):
        self.char_file = char_file
        self.char = _load_char_model(char_file)
        self.cache = {}
        if rollout_data is not None:
            self.motion_files = [rollout_data["file"]]
            self.cache[rollout_data["file"]] = rollout_data
        else:
            self.motion_files = _load_motion_entries(motion_file)
        return

    def get_motion(self, motion_file):
        if motion_file not in self.motion_files:
            raise ValueError("Unknown motion file: {}".format(motion_file))
        if motion_file not in self.cache:
            self.cache[motion_file] = _compute_body_pos(self.char, motion_file)
        return self.cache[motion_file]

    def build_handler(self):
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return

            def _send(self, status, content_type, body):
                if isinstance(body, str):
                    body = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return

            def do_GET(self):
                try:
                    parsed = urlparse(self.path)
                    if parsed.path == "/":
                        self._send(200, "text/html; charset=utf-8", HTML)
                    elif parsed.path == "/api/motions":
                        payload = [
                            {"file": path, "name": _motion_name(path)}
                            for path in viewer.motion_files
                        ]
                        self._send(200, "application/json; charset=utf-8", json.dumps(payload))
                    elif parsed.path == "/api/motion":
                        query = parse_qs(parsed.query)
                        motion_file = query.get("file", [viewer.motion_files[0]])[0]
                        payload = viewer.get_motion(motion_file)
                        self._send(200, "application/json; charset=utf-8", json.dumps(payload))
                    else:
                        self._send(404, "text/plain; charset=utf-8", "Not found")
                except Exception as e:
                    self._send(500, "text/plain; charset=utf-8", str(e))
                return

        return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char_file", default="data/assets/go2/go2.xml")
    parser.add_argument("--motion_file", default="data/motions/go2/go2_apex_trot_clean.pkl")
    parser.add_argument("--model_file", default="")
    parser.add_argument("--env_config", default="data/envs/smp_go2_trot_track_env.yaml")
    parser.add_argument("--agent_config", default="output/smp_go2_trot_track_reward/agent_config.yaml")
    parser.add_argument("--engine_config", default="data/engines/isaac_gym_engine.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout_seconds", type=float, default=10.0)
    parser.add_argument("--stop_on_done", action="store_true")
    parser.add_argument("--master_port", type=int, default=6123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    rollout_data = None
    if args.model_file != "":
        print("Collecting policy rollout from {}...".format(args.model_file), flush=True)
        rollout_data = _collect_policy_rollout(args)
        info = rollout_data["rollout"]
        print(
            "Collected {} frames, mean_reward={:.4f}, sum_reward={:.4f}, done_step={}".format(
                info["num_frames"],
                info["mean_reward"],
                info["sum_reward"],
                info["done_step"],
            ),
            flush=True,
        )

    viewer = ViewerServer(args.char_file, args.motion_file, rollout_data=rollout_data)
    server = ThreadingHTTPServer((args.host, args.port), viewer.build_handler())
    url_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print("Go2 web viewer: http://{}:{}/".format(url_host, args.port), flush=True)
    print("Motions:", flush=True)
    for path in viewer.motion_files:
        print("  - {}".format(path), flush=True)
    server.serve_forever()
    return


if __name__ == "__main__":
    main()
