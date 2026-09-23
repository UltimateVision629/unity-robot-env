using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityRobotEnv.Core;
using Mujoco;
using UnityEngine;

namespace UnityRobotEnv.Networking
{
    /// <summary>
    /// Lightweight TCP server that exposes LiberoEnvironment for RL/BC training.
    /// Runs on port 5556, separate from JoyConReceiver (5555).
    /// JSON-line protocol: {"cmd":"reset"} / {"cmd":"step","action":[...]} / {"cmd":"get_obs"}
    /// </summary>
    public class TrainingServer : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AutoCreate()
        {
            if (FindObjectOfType<TrainingServer>() != null) return;
            var go = new GameObject("TrainingServer");
            DontDestroyOnLoad(go);
            go.AddComponent<TrainingServer>();
        }

        [Header("TCP Settings")]
        public int ListenPort = 5556;
        public bool AutoStart = true;

        [Header("Task (used without LiberoEnvironment)")]
        public string LanguageInstruction = "pick up the red block";

        [Header("References")]
        public LiberoEnvironment Env;

        [Tooltip("Optional MJPEG viewer. Created automatically; inert until enabled.")]
        public MjpegStreamer Streamer;

        private TcpListener _listener;
        private TcpClient _client;
        private Thread _recvThread;
        private readonly ConcurrentQueue<string> _msgQueue = new ConcurrentQueue<string>();
        private readonly ConcurrentQueue<string> _replyQueue = new ConcurrentQueue<string>();
        private volatile bool _running;
        private volatile bool _updateReady;
        private readonly object _sendLock = new object();

        // Remote mirror protocol: the remote simulation is authoritative and
        // streams this raw MuJoCo state to a windowed local player.  Keep this
        // independent of observations: it must never render or encode images.
        private const int SimStateProtocol = 1;
        private long _episodeId;
        private long _simStateSequence;

        public bool IsConnected { get; private set; }

        void Start()
        {
            if (Env == null)
                Env = FindObjectOfType<LiberoEnvironment>();

            // The optional MJPEG viewer lives on its own GameObject and is inert unless
            // MjpegStreamer.EnableStream is set (or stream_start is sent at runtime).
            if (Streamer == null)
            {
                Streamer = FindObjectOfType<MjpegStreamer>();
                if (Streamer == null)
                {
                    var sgo = new GameObject("MjpegStreamer");
                    DontDestroyOnLoad(sgo);
                    Streamer = sgo.AddComponent<MjpegStreamer>();
                }
            }

            if (AutoStart)
                StartServer();
        }

        public void StartServer()
        {
            if (_running) return;
            _running = true;
            _recvThread = new Thread(ServerLoop)
            {
                IsBackground = true,
                Name = "TrainingServer"
            };
            _recvThread.Start();
        }

        private void ServerLoop()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, ListenPort);
                _listener.Start();
                Debug.Log($"[TrainingServer] Listening on 127.0.0.1:{ListenPort}");

                // Wait for Update() to confirm the main thread is running
                while (_running && !_updateReady)
                    Thread.Sleep(50);
                Debug.Log("[TrainingServer] Main thread ready, accepting clients.");

                while (_running)
                {
                    if (_client == null || !_client.Connected)
                    {
                        try
                        {
                            _client = _listener.AcceptTcpClient();
                            _client.NoDelay = true;
                            IsConnected = true;
                            Debug.Log("[TrainingServer] Python client connected.");
                        }
                        catch (SocketException)
                        {
                            if (!_running) break;
                            Thread.Sleep(500);
                            continue;
                        }
                    }

                    try
                    {
                        using (var stream = _client.GetStream())
                        using (var reader = new StreamReader(stream, Encoding.UTF8))
                        {
                            while (_running && _client.Connected)
                            {
                                string line = reader.ReadLine();
                                if (line == null) break;
                                _msgQueue.Enqueue(line);

                                // Wait for reply from main thread, then send
                                string reply = SpinWaitForReply(30000);
                                if (reply != null)
                                {
                                    lock (_sendLock)
                                    {
                                        byte[] payload = Encoding.UTF8.GetBytes(reply + "\n");
                                        stream.Write(payload, 0, payload.Length);
                                        stream.Flush();
                                    }
                                }
                            }
                        }
                    }
                    catch (IOException)
                    {
                        Debug.LogWarning("[TrainingServer] Python client disconnected.");
                    }

                    CleanupClient();
                    IsConnected = false;
                }
            }
            catch (SocketException ex)
            {
                Debug.LogError($"[TrainingServer] Socket error: {ex.Message}");
            }
            finally
            {
                CleanupClient();
                _listener?.Stop();
            }
        }

        private string SpinWaitForReply(int timeoutMs)
        {
            int waited = 0;
            while (waited < timeoutMs)
            {
                if (_replyQueue.TryDequeue(out string reply))
                    return reply;
                Thread.Sleep(10);
                waited += 10;
            }
            return "{\"error\":\"timeout\"}";
        }

        private void CleanupClient()
        {
            try { _client?.Close(); } catch { }
            _client = null;
        }

        void Update()
        {
            _updateReady = true;

            // Optional MJPEG: renders + encodes only while a browser is attached, so
            // this is a no-op in a normal headless inference run.
            Streamer?.StreamTick();

            while (_msgQueue.TryDequeue(out string msg))
            {
                string reply = ProcessMessage(msg);
                _replyQueue.Enqueue(reply);
            }
        }

        private string ProcessMessage(string json)
        {
            try
            {
                string cmd = ExtractString(json, "cmd");

                switch (cmd)
                {
                    case "reset":
                        return HandleReset();
                    case "step":
                        float[] action = ExtractFloatArray(json, "action");
                        return HandleStep(action);
                    case "get_obs":
                        return HandleGetObs();
                    case "get_task":
                        return HandleGetTask();
                    case "get_sim_state":
                        return HandleGetSimState();
                    case "set_preview":
                        // 推理/训练时隐藏右上角腕部小窗（采集时默认显示，不调用）
                        bool show = !string.Equals(ExtractString(json, "show"), "false");
                        var oc = FindObjectOfType<ObservationCollector>();
                        if (oc != null) oc.SetWristPreview(show);
                        return $"{{\"ok\":true,\"show\":{(show ? "true" : "false")}}}";

                    // ── optional MJPEG viewer (see MjpegStreamer) ─────────────
                    case "stream_start":
                        if (Streamer == null) return "{\"error\":\"no streamer\"}";
                        {
                            // Optional knobs for watching DURING inference, when the
                            // per-tick cost must be minimal:
                            //   {"cmd":"stream_start","fps":5,"cam":"agent"}
                            // fps caps main-thread encodes; cam=agent skips the second
                            // camera render entirely (halves the per-tick cost).
                            string sfps = ExtractString(json, "fps");
                            if (float.TryParse(sfps, out float f) && f >= 1f && f <= 60f)
                                Streamer.StreamFps = (int)f;
                            string scam = ExtractString(json, "cam");
                            if (!string.IsNullOrEmpty(scam))
                                Streamer.AgentOnly = (scam == "agent");
                            string sscale = ExtractString(json, "scale");
                            if (int.TryParse(sscale, out int sc) && sc >= 1 && sc <= 4)
                                Streamer.Downscale = sc;
                        }
                        bool ok = Streamer.StartStream();
                        Streamer.ApplySettings(Streamer.StreamFps, Streamer.AgentOnly);
                        return $"{{\"ok\":{(ok ? "true" : "false")},"
                             + $"\"port\":{Streamer.StreamPortActual},"
                             + $"\"fps\":{Streamer.StreamFps},"
                             + $"\"agent_only\":{(Streamer.AgentOnly ? "true" : "false")}}}";
                    case "stream_stop":
                        Streamer?.StopStream();
                        return "{\"ok\":true}";
                    case "stream_status":
                        if (Streamer == null) return "{\"ok\":true,\"enabled\":false}";
                        return $"{{\"ok\":true,\"enabled\":"
                             + $"{(Streamer.IsEnabled ? "true" : "false")},"
                             + $"\"clients\":{Streamer.ClientCount},"
                             + $"\"port\":{Streamer.StreamPortActual},"
                             + $"\"has_frame\":{(Streamer.HasFrame ? "true" : "false")}}}";

                    default:
                        return $"{{\"error\":\"unknown cmd: {cmd}\"}}";
                }
            }
            catch (Exception ex)
            {
                return $"{{\"error\":\"{ex.Message}\"}}";
            }
        }

        private string HandleReset()
        {
            _stepCount = 0;
            _episodeId++;
            _simStateSequence = 0;

            if (Env != null)
                return ObsToJson(Env.ResetEnvironment());

            ResetScene();
            return ObsToJson(CollectObsFallback());
        }

        /// <summary>
        /// MuJoCo 场景重置 + 双臂回 home（与 TCP reset 命令同一逻辑，供 UI 按钮调用）。
        /// mj_resetData 恢复初始状态；OnSceneReset 立即把双臂重定位到 home keyframe
        /// 并丢弃缓存的 Joy-Con pose——否则 MjJoyConController 会继续把上一集最后的
        /// 关节写进 ctrl，重置后机械臂回到旧位置。
        /// </summary>
        public void ResetScene()
        {
            // In read-only mirror mode the remote player owns reset/random
            // placement.  Forward UI/TCP reset intent instead of mutating this
            // local display scene.
            var mirror = FindObjectOfType<RemoteMirrorClient>();
            if (mirror != null && mirror.IsMirrorActive)
            {
                mirror.RequestReset();
                Debug.Log("[TrainingServer] Forwarded reset request to remote mirror authority.");
                return;
            }
            if (MjScene.InstanceExists)
            {
                unsafe
                {
                    MujocoLib.mj_resetData(MjScene.Instance.Model, MjScene.Instance.Data);
                    RandomizeBlocks();
                }
                var mjCtrl = FindObjectOfType<MjJoyConController>();
                if (mjCtrl != null)
                    mjCtrl.OnSceneReset();
                Debug.Log("[TrainingServer] MuJoCo scene reset to initial state.");
            }
        }

        // ── 方块随机摆放（2026-08-11）─────────────────────────────────
        // 每次 reset 后把三个方块随机放到机械臂可达范围内：
        //   红 → 右臂可达圈、蓝 → 左臂可达圈、绿 → 双臂可达交叠区（保证
        //   "green with left/right arm" 两个任务在任意一次 reset 都可行）。
        // 约束（俯视 x-y 平面，单位 m）：
        //   · 与指配基座水平距离 ∈ [0.28, 0.46]（上限 0.46 = 数据实测最远抓取
        //     0.461m（红块）；0.28 下限避开基座/折叠臂结构）；
        //   · 离另一基座 ≥0.28（红离 L 基座、蓝离 R 基座）；
        //   · 方块中心完全在桌面上（x∈[-0.475, 0.675], |y|≤0.475，已扣 0.025
        //     半宽；桌台中心 2026-08-11 前移20cm 后移10cm → x=+0.10 → 桌面 x∈[-0.50, 0.70]）；
        //   · 方块间中心距 ≥0.10（5cm 方块 + 5cm 间隙）；
        //   · z=0.03；四元数直立 + 随机 yaw（±BLOCK_YAW_MAX，绕 Z 轴，数据增强）。
        // 注（2026-08-11）：HomeArms 已把双臂瞬移 home（qpos 直写），不存在
        // 趴平臂横扫的过渡状态，故不再排除基座前方走廊（旧版曾排除 dx∈[0.26,0.45]、
        // |dy|<0.10 的窄条——实测 HomeArms 后无趴平态，且 qpos=0 趴平臂模拟
        // 本就不碰动方块）。
        // 拒绝采样 3000 次/方块，整组重试 3 轮；仍失败则保留 XML 默认位置。
        private const float BLOCK_REACH_MIN = 0.28f;
        private const float BLOCK_REACH_MAX = 0.46f;
        private const float BLOCK_SEP = 0.10f;
        private const float BLOCK_TABLE_X_MIN = -0.50f + 0.025f;
        private const float BLOCK_TABLE_X_MAX = 0.70f - 0.025f;
        private const float BLOCK_TABLE_Y = 0.50f - 0.025f;
        private const float BLOCK_Z = 0.03f;
        private const float BLOCK_YAW_MAX = 30f;   // 方块初始化随机 yaw（度）——数据增强（2026-09-01）

        private static readonly (string name, int baseIdx, bool bothArms)[] _blockSpec = {
            ("red_block_1", 0, false),    // 右臂可达圈
            ("blue_block_1", 1, false),   // 左臂可达圈
            ("green_block_1", 0, true),   // 双臂都可达
        };
        private static readonly (float x, float y)[] _armBases = {
            (-0.70f, -0.20f),   // R 基座
            (-0.70f,  0.20f),   // L 基座
        };

        private bool _singleArm;          // 单臂场景（无 L_ 关节）→ 三块全指右臂
        private bool _singleArmChecked;

        /// <summary>
        /// 检测当前模型是否单臂（无 L_Rotation 关节）——libero_put_block_in_box 场景
        /// 只保留右臂。单臂时 _blockSpec 的臂指派与"离另一基座"约束不再适用。
        /// </summary>
        private unsafe bool DetectSingleArm()
        {
            if (_singleArmChecked) return _singleArm;
            _singleArmChecked = true;
            _singleArm = false;
            var model = MjScene.Instance.Model;
            if ((IntPtr)model == IntPtr.Zero || model->names == null || model->njnt <= 0) return _singleArm;
            byte* names = (byte*)model->names;
            for (int j = 0; j < model->njnt; j++)
            {
                int adr = model->name_jntadr[j];
                string jname = "";
                for (int k = 0; k < 80; k++)
                {
                    char c = (char)names[adr + k];
                    if (c == '\0') break;
                    jname += c;
                }
                if (jname.StartsWith("L_Rotation")) return _singleArm;   // 有左臂
            }
            _singleArm = true;
            Debug.Log("[TrainingServer] 单臂场景（无 L_ 关节）：三个方块全部指派右臂可达圈");
            return _singleArm;
        }

        private unsafe void RandomizeBlocks()
        {
            var model = MjScene.Instance.Model;
            var data = MjScene.Instance.Data;
            if ((IntPtr)model == IntPtr.Zero || model->names == null) return;

            bool singleArm = DetectSingleArm();   // 单臂场景：三块全指右臂、无另一基座约束

            // 方块 body 前缀匹配（mj_name2id 因 _NNN 后缀失效，同 MjJoyConController）
            // → body_jntadr 找到 freejoint → jnt_qposadr
            var qposAdr = new Dictionary<string, int>();
            byte* names = (byte*)model->names;
            for (int b = 0; b < model->nbody; b++)
            {
                int adr = model->name_bodyadr[b];
                string bname = "";
                for (int k = 0; k < 80; k++)
                {
                    char c = (char)names[adr + k];
                    if (c == '\0') break;
                    bname += c;
                }
                foreach (var spec in _blockSpec)
                {
                    if (qposAdr.ContainsKey(spec.name)) continue;
                    if (bname.StartsWith(spec.name) && model->body_jntnum[b] >= 1)
                        qposAdr[spec.name] = model->jnt_qposadr[model->body_jntadr[b]];
                }
            }
            if (qposAdr.Count != _blockSpec.Length)
            {
                Debug.LogWarning($"[TrainingServer] RandomizeBlocks: 找到 {qposAdr.Count}/{_blockSpec.Length} 个方块，跳过随机化");
                return;
            }

            var positions = new Dictionary<string, (float x, float y)>();
            var placed = new List<(float x, float y)>();
            for (int round = 0; round < 3 && positions.Count < _blockSpec.Length; round++)
            {
                positions.Clear();
                placed.Clear();
                foreach (var spec in _blockSpec)
                {
                    int baseIdx = singleArm ? 0 : spec.baseIdx;   // 单臂场景全指右臂（R 基座）
                    float bx = _armBases[baseIdx].x;
                    float by = _armBases[baseIdx].y;
                    bool found = false;
                    for (int i = 0; i < 3000 && !found; i++)
                    {
                        float ang = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
                        float r = Mathf.Sqrt(UnityEngine.Random.Range(
                            BLOCK_REACH_MIN * BLOCK_REACH_MIN, BLOCK_REACH_MAX * BLOCK_REACH_MAX));
                        float x = bx + r * Mathf.Cos(ang);
                        float y = by + r * Mathf.Sin(ang);
                        if (x < BLOCK_TABLE_X_MIN || x > BLOCK_TABLE_X_MAX || Mathf.Abs(y) > BLOCK_TABLE_Y) continue;
                        if (singleArm)
                        {
                            // 方块只在盒子右端点（y=-0.022）的右侧生成——agentview 画面右
                            // = MuJoCo -Y = 机械臂侧；方块不越过盒子右端（画面左/盒子
                            // 左后侧不生成）。注意：y 方向才是"画面左右"，x 约束会把方块
                            // 推到 agentview 桌边（历史错误约束，勿恢复）
                            if (y > -0.022f) continue;
                            // 排除盒子区域
                            // （XML box_1 中心 (-0.42, 0.02)，外半廓 0.0424 + 块半宽 0.025 + 间隙 0.02）
                            float dB = Mathf.Sqrt((x + 0.42f) * (x + 0.42f) + (y - 0.02f) * (y - 0.02f));
                            if (dB < 0.087f) continue;
                        }
                        else if (spec.bothArms)
                        {
                            // 绿：左臂也要够得到
                            float dL = Mathf.Sqrt((x - _armBases[1].x) * (x - _armBases[1].x)
                                                + (y - _armBases[1].y) * (y - _armBases[1].y));
                            if (dL < BLOCK_REACH_MIN || dL > BLOCK_REACH_MAX) continue;
                        }
                        else
                        {
                            // 离另一基座的结构远一点（红→L 基座，蓝→R 基座）
                            int other = 1 - spec.baseIdx;
                            float dO = Mathf.Sqrt((x - _armBases[other].x) * (x - _armBases[other].x)
                                                + (y - _armBases[other].y) * (y - _armBases[other].y));
                            if (dO < BLOCK_REACH_MIN) continue;
                        }
                        bool sepOk = true;
                        foreach (var p in placed)
                        {
                            float dx = p.x - x, dy = p.y - y;
                            if (dx * dx + dy * dy < BLOCK_SEP * BLOCK_SEP) { sepOk = false; break; }
                        }
                        if (!sepOk) continue;
                        found = true;
                        placed.Add((x, y));
                        positions[spec.name] = (x, y);
                    }
                }
            }

            if (positions.Count == _blockSpec.Length)
            {
                foreach (var kv in positions)
                {
                    int adr = qposAdr[kv.Key];
                    data->qpos[adr + 0] = kv.Value.x;
                    data->qpos[adr + 1] = kv.Value.y;
                    data->qpos[adr + 2] = BLOCK_Z;
                    // 四元数 wxyz：直立 + 随机 yaw（绕 MuJoCo Z 轴转 ±BLOCK_YAW_MAX，
                    // 数据增强：方块姿态不再恒与臂对齐，模型须从图像读姿态）
                    float yaw = UnityEngine.Random.Range(-BLOCK_YAW_MAX, BLOCK_YAW_MAX) * Mathf.Deg2Rad;
                    data->qpos[adr + 3] = Mathf.Cos(yaw * 0.5f);
                    data->qpos[adr + 4] = 0.0f;
                    data->qpos[adr + 5] = 0.0f;
                    data->qpos[adr + 6] = Mathf.Sin(yaw * 0.5f);
                }
                MujocoLib.mj_kinematics(model, data);   // 立即更新几何位置（下一帧也会更新）
                string summary = "";
                foreach (var kv in positions) summary += $"{kv.Key}=({kv.Value.x:F3},{kv.Value.y:F3}) ";
                Debug.Log($"[TrainingServer] Blocks randomized: {summary.Trim()}");
            }
            else
            {
                Debug.LogWarning($"[TrainingServer] RandomizeBlocks 3 轮拒绝采样均失败（{positions.Count}/3），保留 XML 默认位置");
            }
        }

        private int _stepCount = 0;

        private string HandleStep(float[] action)
        {
            if (Env == null)
            {
                // MuJoCo-only mode: apply EEF delta action via IK
                var mjCtrl = FindObjectOfType<MjJoyConController>();
                if (mjCtrl != null && action.Length >= 14)
                {
                    float[] rightAction = new float[7]; System.Array.Copy(action, 0, rightAction, 0, 7);
                    float[] leftAction  = new float[7]; System.Array.Copy(action, 7, leftAction,  0, 7);
                    // DEBUG: log first few steps
                    if (_stepCount < 3)
                        Debug.Log($"[TrainingServer] Step {_stepCount}: R_dx={rightAction[0]:F6} R_dz={rightAction[2]:F6} R_grip={rightAction[6]:F4} L_dx={leftAction[0]:F6}");
                    mjCtrl.ApplyEefDelta(0, rightAction);
                    mjCtrl.ApplyEefDelta(1, leftAction);
                    // Step physics enough times for actuators to track the target
                    if (MjScene.InstanceExists)
                    {
                        unsafe {
                            for (int s = 0; s < 10; s++)
                                MujocoLib.mj_step(MjScene.Instance.Model, MjScene.Instance.Data);
                        }
                    }
                }

                _stepCount++;
                var fallbackObs = CollectObsFallback();
                return "{\"reward\":0.0000,\"done\":false,\"step\":" + _stepCount +
                       ",\"success\":false,\"obs\":" + ObsToJson(fallbackObs) + "}";
            }

            var (obs, reward, done, info) = Env.Step(action);

            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"reward\":{reward:F4},");
            sb.Append($"\"done\":{(done ? "true" : "false")},");
            sb.Append($"\"step\":{info.GetValueOrDefault("step", 0)},");
            sb.Append($"\"success\":{((bool)info.GetValueOrDefault("success", false) ? "true" : "false")},");
            sb.Append("\"obs\":");
            sb.Append(ObsToJson(obs));
            sb.Append("}");
            return sb.ToString();
        }

        private string HandleGetObs()
        {
            if (Env != null)
            {
                var method = typeof(LiberoEnvironment).GetMethod("GatherObservation",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                return ObsToJson((Observation)method.Invoke(Env, null));
            }

            // Fallback: collect observation directly (MuJoCo XML import mode)
            return ObsToJson(CollectObsFallback());
        }

        private Observation CollectObsFallback()
        {
            var collector = FindObjectOfType<ObservationCollector>();
            if (collector == null)
                return new Observation { JointPositions = new float[7], EEFPosition = new float[3], EEFQuaternion = new float[4], GripperQPos = new float[2] };

            var obs = collector.Collect(null, null);

            // Add MuJoCo free-body positions for grasp verification
            unsafe
            {
                if (MjScene.InstanceExists)
                {
                    var model = MjScene.Instance.Model;
                    var data = MjScene.Instance.Data;
                    string[] knownObjects = { "red_block_1", "green_block_1", "blue_block_1" };
                    if (obs.ObjectPositions == null)
                        obs.ObjectPositions = new Dictionary<string, float[]>();
                    foreach (var name in knownObjects)
                    {
                        int bodyId = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_BODY, name);
                        if (bodyId >= 0)
                        {
                            obs.ObjectPositions[name] = new float[] {
                                (float)data->xpos[3 * bodyId],
                                (float)data->xpos[3 * bodyId + 1],
                                (float)data->xpos[3 * bodyId + 2]
                            };
                        }
                    }
                }
            }

            return obs;
        }

        private string HandleGetTask()
        {
            if (Env != null)
            {
                string lang = Env.GetLanguageInstruction();
                return $"{{\"language_instruction\":\"{EscapeJson(lang)}\"}}";
            }
            return $"{{\"language_instruction\":\"{EscapeJson(LanguageInstruction)}\"}}";
        }

        /// <summary>
        /// Export the complete dynamic MuJoCo state for a read-only mirror
        /// player.  This deliberately contains no observation fields and no
        /// rendered image data: qpos covers robot and free-joint objects,
        /// while qvel/act/ctrl preserve the rest of the physical state.
        /// </summary>
        public unsafe string HandleGetSimState()
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null || MjScene.Instance.Data == null)
                return "{\"error\":\"MuJoCo scene is not ready\"}";

            var model = MjScene.Instance.Model;
            var data = MjScene.Instance.Data;
            var sb = new StringBuilder();
            sb.Append("{\"ok\":true");
            sb.Append(",\"protocol\":").Append(SimStateProtocol);
            sb.Append(",\"episode_id\":").Append(_episodeId);
            sb.Append(",\"seq\":").Append(++_simStateSequence);
            sb.Append(",\"sim_time\":").Append(data->time.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            sb.Append(",\"scene_hash\":\"").Append(ComputeSceneHash(model)).Append("\"");
            sb.Append(",\"nq\":").Append(model->nq);
            sb.Append(",\"nv\":").Append(model->nv);
            sb.Append(",\"na\":").Append(model->na);
            sb.Append(",\"nu\":").Append(model->nu);
            sb.Append(",\"qpos\":").Append(DoubleArrayToJson(data->qpos, model->nq));
            sb.Append(",\"qvel\":").Append(DoubleArrayToJson(data->qvel, model->nv));
            sb.Append(",\"act\":").Append(DoubleArrayToJson(data->act, model->na));
            sb.Append(",\"ctrl\":").Append(DoubleArrayToJson(data->ctrl, model->nu));
            sb.Append(",\"joint_qposadr\":[");
            for (int j = 0; j < model->njnt; j++)
            {
                if (j > 0) sb.Append(',');
                sb.Append(model->jnt_qposadr[j]);
            }
            sb.Append("],\"joint_type\":[");
            for (int j = 0; j < model->njnt; j++)
            {
                if (j > 0) sb.Append(',');
                sb.Append(model->jnt_type[j]);
            }
            sb.Append("]");
            sb.Append(",\"joint_names\":").Append(MuJoCoNameArrayToJson(model, MujocoLib.mjtObj.mjOBJ_JOINT, model->njnt));
            sb.Append(",\"actuator_names\":").Append(MuJoCoNameArrayToJson(model, MujocoLib.mjtObj.mjOBJ_ACTUATOR, model->nu));
            sb.Append("}");
            return sb.ToString();
        }

        /// <summary>Fingerprint of the runtime MuJoCo layout used by mirror peers.</summary>
        public unsafe string GetSimSceneHash()
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null) return "";
            return ComputeSceneHash(MjScene.Instance.Model);
        }

        private static unsafe string ComputeSceneHash(MujocoLib.mjModel_* model)
        {
            // FNV-1a is sufficient here: this is a compatibility guard, not a
            // cryptographic signature.  Build-time import order differs between
            // Windows and Linux, so numeric qpos/dof addresses are deliberately
            // excluded.  The wire protocol maps qpos/qvel/ctrl by these stable
            // MuJoCo names before applying a packet on the mirror player.
            ulong hash = 14695981039346656037UL;
            void Mix(int value)
            {
                unchecked
                {
                    hash ^= (uint)value;
                    hash *= 1099511628211UL;
                }
            }
            Mix(model->nq); Mix(model->nv); Mix(model->na); Mix(model->nu); Mix(model->njnt);
            var layout = new List<string>();
            for (int j = 0; j < model->njnt; j++)
                layout.Add($"J:{StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_JOINT, j)}:{model->jnt_type[j]}");
            for (int a = 0; a < model->nu; a++)
                layout.Add($"A:{StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_ACTUATOR, a)}");
            layout.Sort(StringComparer.Ordinal);
            foreach (string entry in layout)
                foreach (char ch in entry) Mix(ch);
            return hash.ToString("x16");
        }

        private static unsafe string StableMuJoCoName(MujocoLib.mjModel_* model, MujocoLib.mjtObj type, int id)
        {
            // Do not call mj_id2name here: this Tuanjie binding marshals the
            // native const char* as an owned string and can corrupt the Player
            // heap on Windows.  Components are already bound to the same ids.
            string name = "";
            if (type == MujocoLib.mjtObj.mjOBJ_JOINT)
            {
                foreach (var joint in UnityEngine.Object.FindObjectsOfType<MjBaseJoint>())
                    if (joint.MujocoId == id) { name = joint.MujocoName ?? ""; break; }
            }
            else if (type == MujocoLib.mjtObj.mjOBJ_ACTUATOR)
            {
                foreach (var actuator in UnityEngine.Object.FindObjectsOfType<MjActuator>())
                    if (actuator.MujocoId == id) { name = actuator.MujocoName ?? ""; break; }
            }
            if (string.IsNullOrEmpty(name)) return $"<missing-{(int)type}-{id}>";
            // Tuanjie appends a numeric instance suffix while it imports an
            // otherwise identical scene.  The source joint names here already
            // carry their semantic number before that suffix (joint_29_7), so
            // remove only the final generated segment.
            int split = name.LastIndexOf('_');
            if (split >= 0 && name.Length - split - 1 >= 1)
            {
                bool digits = true;
                for (int i = split + 1; i < name.Length; i++)
                    if (!char.IsDigit(name[i])) { digits = false; break; }
                if (digits) name = name.Substring(0, split);
            }
            return name;
        }

        private static unsafe string DoubleArrayToJson(double* values, int length)
        {
            var sb = new StringBuilder();
            sb.Append('[');
            for (int i = 0; i < length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(values[i].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            }
            sb.Append(']');
            return sb.ToString();
        }

        private static unsafe string MuJoCoNameArrayToJson(MujocoLib.mjModel_* model, MujocoLib.mjtObj type, int length)
        {
            var sb = new StringBuilder();
            sb.Append('[');
            for (int i = 0; i < length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(StableMuJoCoName(model, type, i).Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('"');
            }
            sb.Append(']');
            return sb.ToString();
        }

        private string ObsToJson(Observation obs)
        {
            var dict = obs.ToPythonDict();
            var sb = new StringBuilder();
            sb.Append("{");

            // agentview image → base64 PNG
            if (dict.TryGetValue("agentview_image", out object imgObj) && imgObj is byte[] imgBytes && imgBytes.Length > 0)
            {
                string b64 = Convert.ToBase64String(EncodePng(imgBytes, obs.ImageWidth, obs.ImageHeight));
                sb.Append($"\"agentview_image\":\"{b64}\",");
                sb.Append($"\"image_width\":{obs.ImageWidth},");
                sb.Append($"\"image_height\":{obs.ImageHeight},");
            }

            // eye_in_hand image (optional)
            if (dict.TryGetValue("eye_in_hand_image", out object wristObj) && wristObj is byte[] wristBytes && wristBytes.Length > 0)
            {
                string b64 = Convert.ToBase64String(EncodePng(wristBytes, obs.ImageWidth, obs.ImageHeight));
                sb.Append($"\"eye_in_hand_image\":\"{b64}\",");
            }

            // proprioception
            sb.Append("\"robot0_joint_pos\":");
            sb.Append(FloatArrayToJson(obs.JointPositions));
            sb.Append(",\"robot0_eef_pos\":");
            sb.Append(FloatArrayToJson(obs.EEFPosition));
            sb.Append(",\"robot0_eef_quat\":");
            sb.Append(FloatArrayToJson(obs.EEFQuaternion));
            sb.Append(",\"robot0_gripper_qpos\":");
            sb.Append(FloatArrayToJson(obs.GripperQPos));

            // robot_1 (left arm) proprioception
            sb.Append(",\"robot1_joint_pos\":");
            sb.Append(FloatArrayToJson(obs.JointPositions1));
            sb.Append(",\"robot1_eef_pos\":");
            sb.Append(FloatArrayToJson(obs.EEFPosition1));
            sb.Append(",\"robot1_eef_quat\":");
            sb.Append(FloatArrayToJson(obs.EEFQuaternion1));
            sb.Append(",\"robot1_gripper_qpos\":");
            sb.Append(FloatArrayToJson(obs.GripperQPos1));

            // object states
            if (obs.ObjectPositions != null && obs.ObjectPositions.Count > 0)
            {
                sb.Append(",\"object_positions\":{");
                bool first = true;
                foreach (var kv in obs.ObjectPositions)
                {
                    if (!first) sb.Append(",");
                    sb.Append($"\"{EscapeJson(kv.Key)}\":{FloatArrayToJson(kv.Value)}");
                    first = false;
                }
                sb.Append("}");
            }

            sb.Append("}");
            return sb.ToString();
        }

        private static byte[] EncodePng(byte[] rawRgb, int width, int height)
        {
            // rawRgb is RGB24 from Texture2D.GetRawTextureData()
            // Flip vertically then encode as PNG via Texture2D
            var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(rawRgb);
            tex.Apply();
            byte[] png = tex.EncodeToPNG();
            Destroy(tex);
            return png;
        }

        private static string FloatArrayToJson(float[] arr)
        {
            if (arr == null) return "[]";
            var sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < arr.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(arr[i].ToString("F6"));
            }
            sb.Append("]");
            return sb.ToString();
        }

        private static string ExtractString(string json, string key)
        {
            string search = $"\"{key}\"";
            int idx = json.IndexOf(search);
            if (idx < 0) return null;

            int colon = json.IndexOf(':', idx + search.Length);
            if (colon < 0) return null;

            // skip whitespace after colon
            int start = colon + 1;
            while (start < json.Length && json[start] == ' ')
                start++;

            if (start >= json.Length)
                return "";

            // if quoted, find closing quote
            if (json[start] == '\"')
            {
                start++; // skip opening quote
                int end = json.IndexOf('\"', start);
                if (end < 0) end = json.Length;
                return json.Substring(start, end - start);
            }

            // unquoted → read until comma or }
            int end2 = json.IndexOfAny(new[] { ',', '}' }, start);
            if (end2 < 0) end2 = json.Length;
            return json.Substring(start, end2 - start).Trim();
        }

        private static float[] ExtractFloatArray(string json, string key)
        {
            string search = $"\"{key}\"";
            int idx = json.IndexOf(search);
            if (idx < 0) return new float[0];

            int bracket = json.IndexOf('[', idx + search.Length);
            int end = json.IndexOf(']', bracket);
            if (bracket < 0 || end < 0) return new float[0];

            string inner = json.Substring(bracket + 1, end - bracket - 1);
            string[] parts = inner.Split(',');
            float[] result = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                float.TryParse(parts[i].Trim(), out result[i]);
            return result;
        }

        private static string EscapeJson(string s)
        {
            return s?.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n") ?? "";
        }

        void OnDestroy()
        {
            _running = false;
            CleanupClient();
            _listener?.Stop();
            _recvThread?.Join(2000);
        }
    }
}
