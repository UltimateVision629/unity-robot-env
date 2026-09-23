# UnityRobotEnv

A Unity + MuJoCo environment for **dual-arm SO100 robot teleoperation and data collection**:
Joy-Con teleop → MuJoCo physics in Unity → (observation, action) trajectories → VLA training.

Not a LIBERO product. This project began as a MuJoCo-in-Unity integration and grew into a
standalone teleoperation/collection rig; the LIBERO-derived parts are limited and attributed
in [`NOTICE.md`](NOTICE.md).

| | |
|---|---|
| Simulator | Unity (Tuanjie 2022.3) + MuJoCo Unity plugin 3.2.4 |
| Robot | SO-ARM100 / SO100, single right arm in the `put_in_box` scene |
| Teleop | Joy-Con over a Python client → `JoyConReceiver` (TCP :5555) |
| Data | `TrainingServer` (TCP :5556) serves `get_obs` / `step` / `reset` / `get_task` |
| Recorded demos | 101 successful episodes, 38,892 frames |

## Project Structure

```
Assets/UnityRobotEnv/
├── assets/bddl_files/           BDDL task definitions (from upstream LIBERO)
├── Scripts/
│   ├── BDDL/
│   │   ├── BDDLToken.cs         Token types for BDDL lexer
│   │   ├── Tokenizer.cs         Lisp-style tokenizer
│   │   ├── AST.cs               S-Expression AST (SAtom, SList)
│   │   ├── SExprParser.cs       Token stream to S-Expression parser
│   │   ├── BDDLParser.cs        S-Expression to BDDLProblem mapper
│   │   ├── ProblemModel.cs      Strongly-typed BDDL data structures
│   │   └── BDDLParseException.cs
│   └── Core/
│       ├── AssetDatabase.cs     Asset path resolution
│       ├── SceneBuilder.cs      Scene construction from BDDL (fixtures + objects + placement)
│       ├── LiberoEnvironment.cs Main environment loop (step/reset/check_success)
│       ├── ObservationCollector.cs Camera & proprioception data collection
│       ├── ObjectState.cs       Object position/rotation query wrapper
│       ├── RegionSampler.cs     Random sampling within BDDL-defined regions
│       └── FrankaPandaController.cs Robot IK and action interface
└── Tests/
    ├── BDDLParserTests.cs       BDDL parsing unit tests
    └── LiberoEnvironmentTests.cs Environment integration tests
```

## MVP Scope

- [x] BDDL file parsing (Lisp-like DSL)
- [x] Scene construction from BDDL (fixtures, objects, placement regions)
- [x] Random object placement via region samplers
- [x] Step/reset/check_success environment loop
- [x] Observation collection (camera images + proprioception)
- [x] Goal success checking (On predicate)
- [ ] Full Franka Panda IK solver (currently simplified)
- [ ] MuJoCo Unity plugin integration
- [ ] gRPC/Python communication for training

## Getting Started

1. Open the project in Unity 2022.3 LTS or later
2. Run `Window > General > Test Runner` to execute unit tests
3. Add a `LiberoEnvironment` component to a GameObject
4. Set `BDDLFilePath` to a .bddl file path
5. Press Play to initialize the scene

## License

MIT - See LICENSE file

## Replay & Validation

回放录制的 demo（开环执行）和离线验证：

```bash
conda activate lerobot-kin

# 在线回放（需 Unity Play 模式运行中）
python test/replay_action_via_ik.py --traj ../DatasetsCollector/demos/episode_0001_pick_up_the_red_block_20260728_220541_success.npz

# 离线批量验证
python test/validate_replay_offline.py --dir ../DatasetsCollector/demos

# 验证 MuJoCo FK 与记录 EEF 的物理一致性
python test/verify_eef_match.py
```

### IK 后端配置

`arm_ik.py` 提供统一的 Strategy 模式接口，collect / replay / inference 三端共用 `--ik-backend` 切换：

```bash
# 默认 mujoco（与 Unity 同一物理模型，FK 误差 ~4.5e-6 m）
python test/replay_action_via_ik.py --traj <file.npz>

# 原 lerobot 后端（C 扩展，与 MuJoCo 差 30-50cm——旧行为）
python test/replay_action_via_ik.py --traj <file.npz> --ik-backend lerobot
```

| 后端 | IK 方法 | 模型一致性 | 适用 |
|------|---------|-----------|------|
| `mujoco` (默认) | MuJoCo FK + scipy least_squares + Tikhonov 正则 | ✅ 同一模型 | 回放 / 推理 |
| `lerobot` | C 扩展 fknm.pyd | ❌ 差 30-50cm（回放失败根因） | 采集保留兼容 |
| `placo` | placo + URDF | ⚠️ URDF ≠ XML | stub |

**回放目标模式**（`--ik-target`）：`eef` 直接用 `.npz` 记录的 MuJoCo EEF 观测（误差 2-7mm，仅 mujoco）；`pose` 走链式翻译（所有后端通用，误差 2-4cm rate-limit 滞后）；`auto` 自动检测 obs 有效性切换（默认）。

## 远程查看画面（可选 MJPEG，默认关闭）

场景跑在远程 GPU 机器上时，本机需要看到画面。`MjpegStreamer`
（`Assets/UnityRobotEnv/Scripts/Networking/MjpegStreamer.cs`）把渲染好的 agentview 以
**MJPEG over HTTP** 推给浏览器，本机**不需要装任何东西、不需要跑 viewer 脚本**。

### 为什么不用 `get_obs` 轮询取图

观测协议（:5556）本身就能带图，但每次调用都要在**主线程**重新 PNG 编码、再用 base64
膨胀 33%。一个 viewer 挂在它上面会和推理/控制循环抢主线程。MJPEG 走推流 + JPEG，
既便宜又与主循环解耦。实测权衡：

| 方式 | 每帧数据 | 主线程编码 | 连接开销 |
|---|---|---|---|
| 轮询 `get_obs` | ~270 KB（PNG + base64） | **6–20 ms** | 每帧新建 TCP |
| **MJPEG** | ~15–40 KB（JPEG） | **1–2 ms**，且仅在有观众时 | 单条长连接 |

### 三层开关（默认全关）

```csharp
[Header("MJPEG (optional, default OFF)")]
public bool   EnableStream = false;      // 关 = 连监听都不建
public int    StreamPort   = 5560;       // 独立端口，不碰 5555 / 5556
public string BindAddress  = "0.0.0.0";  // 改 127.0.0.1 可禁止远程访问
public int    StreamFps    = 15;
public int    JpegQuality  = 75;
```

| 状态 | 主线程开销 | 网络开销 |
|---|---|---|
| `EnableStream = false` | **0** | **0** |
| 开着，但没有观众 | **0** —— `SubmitFrames()` 直接 return | **0** |
| 有观众连接 | 1–2 ms/帧，按 `StreamFps` 限速 | 15–40 KB/帧 |

**关键**：没有人看的时候**一帧都不编码**，所以不看视频的推理运行完全不受影响。

### 启用 / 关闭

**方式一：运行时命令**（推荐 —— 不需要重启，推理脚本自己控制）

通过现有的 `:5556` JSON-line 协议发送，**原有协议完全不变**：

```json
{"cmd":"stream_start"}    ->  {"ok":true,"port":5560}
{"cmd":"stream_stop"}     ->  {"ok":true}
{"cmd":"stream_status"}   ->  {"ok":true,"enabled":true,"clients":1,"port":5560,"has_frame":true}
```

Python 例子：

```python
import json, socket

def stream_cmd(cmd, port=5556):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((json.dumps({"cmd": cmd}) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        buf += s.recv(65536)
    s.close()
    return json.loads(buf.decode())

stream_cmd("stream_start")     # 想看画面时
...
stream_cmd("stream_stop")      # 不看了，彻底停掉编码
```

**方式二：启动即开**（把 `EnableStream` 改成 `true` 后重新构建）

### 看画面

**本地 player**（播放机就在本机）：浏览器直接打开

```
http://127.0.0.1:5560/
```

**远程 player（如 AutoDL）**：容器端口默认不对外暴露，用 **SSH 隧道**最省事
（不用改 AutoDL 的自定义服务配置）：

```bash
# 在本地开一条隧道（-N 只转发不开 shell）
ssh -N -L 5560:127.0.0.1:5560 -L 5556:127.0.0.1:5556 <user>@<host> -p <port>
# 然后本机浏览器打开
#   http://127.0.0.1:5560/
```

⚠️ 走隧道时**不能用"本地端口是否监听"判断流的状态** —— 隧道本身一直占着那个端口。
用 `stream_status` 判断，或给测试脚本加 `--via-tunnel`：

```bash
python tools/check_mjpeg_stream.py --host 127.0.0.1 --via-tunnel
```

可用地址（两种拓扑都一样）：

```
http://<host>:5560/                    # 内嵌页面，同时显示 agentview 和 eye_in_hand
http://<host>:5560/stream              # 只要 agentview 的 MJPEG 流
http://<host>:5560/stream?cam=wrist    # 只要 eye_in_hand
http://<host>:5560/status              # JSON 状态
```

命令行验证（不需要浏览器）：

```bash
curl -s --max-time 3 http://127.0.0.1:5560/status
curl -s --max-time 3 http://127.0.0.1:5560/stream | head -c 200 | xxd | head -5
```

### 实现说明（两个踩过的坑）

1. **没有用 `HttpListener`** —— Windows 上需要 URL ACL（非管理员的 player 会失败），
   而且 `HttpListenerResponse` 会在显式 `Content-Length` 之上再插自己的分块编码，
   浏览器会拒绝这样的 multipart MJPEG 流。改用**原始 `TcpListener` + 最小 HTTP/1.1**，
   约 60 行，无平台差异。
2. **限速用 `Time.unscaledTimeAsDouble`**，不是 `Time.frameCount` —— 后者在帧率
   变化时实际 fps 会跟着变。

## 无头 Linux 运行（远程 GPU 机器）

在**没有显示器**的云 GPU 机器上运行 Linux player，需要两件事，缺一个就段错误：

```bash
cd <player 目录>
xvfb-run -a -s "-screen 0 800x600x24" ./unity-robot-env -force-glcore -logFile /tmp/unity.log &
```

| 缺少 | 症状 |
|---|---|
| `xvfb-run` | `Error getting num native displays: Video subsystem has not been initialized` → `No available video device` → 段错误 |
| `-force-glcore` | Vulkan 初始化成功后仍在主循环段错误（`signo:11 addr:0x268`） |

**2026-09-21 在 AutoDL（RTX 4080 SUPER / Ubuntu 22.04）实测：**

| 组合 | 结果 |
|---|---|
| 默认（Vulkan，无显示） | ❌ `No available video device` → 段错误 |
| xvfb + Vulkan | ❌ 仍段错误 |
| **xvfb + `-force-glcore`** | ✅ **成功**，读到真实 agentview（std 74.8） |

因此 `BuildLinux.cs` 里图形后端顺序已改为 **`{ OpenGLCore, Vulkan }`**
（原来是 Vulkan 优先）。注意 `-force-glcore` 是运行时参数，构建时顺序不保证等价，
所以**启动命令里保留它**最稳。

系统依赖（Ubuntu）：

```bash
apt install -y xvfb libegl1 libgl1
```

### 场景是怎么进 player 的

构建脚本（`Assets/Editor/BuildLinux.cs` / `BuildWindows.cs`）与你在编辑器里手动
**Import MuJoCo Scene** 用**同一个导入器**，只是全自动化：

```csharp
var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, ...);
var root  = new MjImporterWithAssets().ImportFile(
                "Assets/UnityRobotEnv/assets/robots/so100_mjcf/libero_put_block_in_box.xml");
EditorSceneManager.SaveScene(scene, "Assets/Local/Build/libero_put_block_in_box.build.scene");
BuildPipeline.BuildPlayer(new BuildPlayerOptions { scenes = new[] { SceneAsset }, ... });
```

运行时：player 加载该场景 → `MjComponent` 层级已在场景中 →
`MjScene` 懒创建 + `FindObjectsOfType` 收集 → 物理开始跑。

两点注意：

- `DefaultGameObjects` 自动带 **Main Camera + Directional Light**；**缺光 agentview 全黑**
- 构建固定使用上面那个 xml 路径。**改了编辑器里的场景或别的 xml，player 不会自动跟着
  变 —— 必须重新构建**

## 资产获取（重要）

本仓库**不包含** LIBERO 自带的第三方物体资产库。若要跑 `put_in_box` 之外的
LIBERO 任务，请自行从上游获取后放到对应路径：

```bash
# 从上游 LIBERO 取得后，放到：
Assets/UnityRobotEnv/assets/turbosquid_objects/      # TurboSquid 购买的模型
Assets/UnityRobotEnv/assets/stable_hope_objects/     # STABLeHOpe 物体库
Assets/UnityRobotEnv/assets/stable_scanned_objects/  # 扫描实物库
Assets/UnityRobotEnv/assets/articulated_objects/     # 铰接物体（橱柜/微波炉等）
Assets/UnityRobotEnv/assets/textures/                # 上述物体所用贴图
```

上游地址：https://github.com/Lifelong-Robot-Learning/LIBERO

**排除原因与影响范围**：

- 这些资产的再分发条款不明确或受限（TurboSquid 模型的再分发取决于原始购买许可），
  且它们不是本移植工程的产物。
- **本工程的 `put_in_box` 场景不依赖它们**：只用到
  `assets/robots/so100_mjcf/`（SO100 模型 + STL，45 个文件 / 约 3 MB）以及 XML 里
  定义的程序化几何（桌面、盒子、三个方块都是 `<geom type="box">`）。实测
  `libero_put_block_in_box.scene` 对这些目录名的引用次数均为 0。
- 也就是说，**克隆后直接跑 `put_in_box` 无需任何额外下载**。

`.gitignore` 已包含这些目录的排除规则。注意 `.gitignore` 只对未跟踪文件生效，
这些目录此前已被跟踪，因此索引中也已用 `git rm -r --cached` 移除。

## MuJoCo Unity 插件（依赖说明）

本工程用 MuJoCo 的 Unity 插件把 MJCF 导入场景、驱动仿真与观测采集。该插件
（Apache-2.0，来自 https://github.com/google-deepmind/mujoco 的 `unity/` 目录，
版本 **3.2.4**）以**内嵌包**形式随仓库分发：

```
Packages/org.mujoco/           # Runtime 62 个 + Editor 14 个 C# 脚本
Packages/org.mujoco/mujoco.dll # 原生库，与 Assets/mujoco.dll 版本对齐
Packages/manifest.json         # 依赖清单（必须入库，否则包无法解析）
```

插件自带的 `Tests/` 已移除（需要完整 NUnit，内嵌使用不需要）。

**注意 `Packages/` 必须在版本控制中。** 本工程早期因 `.gitignore` 里一条无锚定的
`packages/` 规则（Windows 下大小写不敏感）误把整个 `Packages/` 排除，导致新克隆的
仓库既没有依赖清单也没有插件，编辑器里看不到 MuJoCo 导入菜单。该规则已移除，
`Packages/` 现已入库。

## 第三方归属

第三方组件的来源与许可见 [`NOTICE.md`](NOTICE.md)。

## References

- Original LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- robosuite: https://robosuite.ai
- BDDL: https://github.com/StanfordVL/bddl
- MuJoCo: https://github.com/google-deepmind/mujoco
