using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Mujoco;
using UnityEngine;

namespace UnityRobotEnv.Networking
{
    /// <summary>
    /// Read-only player for a remote, authoritative Unity/MuJoCo simulation.
    /// Enable from a Windows player with:
    ///   -remote-mirror-host 127.0.0.1 -remote-mirror-port 7001
    /// The endpoint is normally an SSH local-forward.  Only numeric MuJoCo
    /// state snapshots are accepted; model actions and images stay remote.
    /// </summary>
    public class RemoteMirrorClient : MonoBehaviour
    {
        private const int Protocol = 1;
        private const float DefaultSnapshotPeriod = 1f / 30f;

        [Serializable]
        private class WireMessage
        {
            public string type;
            public string error;
            public int protocol;
            public string scene_hash;
            public long episode_id;
            public long seq;
            public double sim_time;
            public int nq, nv, na, nu;
            public double[] qpos, qvel, act, ctrl;
            public int[] joint_qposadr, joint_type;
            public string[] joint_names, actuator_names;
        }

        [Serializable]
        private class HelloMessage
        {
            public string type = "hello";
            public int protocol = Protocol;
            public string scene_hash;
        }

        [Serializable]
        private class ResetMessage
        {
            public string type = "reset_request";
        }

        private readonly ConcurrentQueue<WireMessage> _received = new ConcurrentQueue<WireMessage>();
        private readonly ConcurrentQueue<string> _outbound = new ConcurrentQueue<string>();
        private Thread _thread;
        private volatile bool _running;
        private volatile bool _connected;
        private string _host;
        private int _port;
        private string _localSceneHash;
        private bool _mirrorEnabled;
        private WireMessage _from;
        private WireMessage _to;
        private float _blendStarted;
        private float _blendDuration = DefaultSnapshotPeriod;
        private long _episode = -1;
        private long _lastSequence = -1;
        // The Linux and Windows importers may assign different numeric joint
        // indices.  These maps convert one received packet into local order.
        private int[] _remoteJointToLocal;
        private int[] _remoteActuatorToLocal;
        private string _status = "Waiting for remote mirror connection";

        public bool IsMirrorActive => _mirrorEnabled;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AutoCreate()
        {
            if (!TryReadArgument("-remote-mirror-port", out _)) return;
            if (FindObjectOfType<RemoteMirrorClient>() != null) return;
            var go = new GameObject("RemoteMirrorClient");
            DontDestroyOnLoad(go);
            go.AddComponent<RemoteMirrorClient>();
        }

        private void Start()
        {
            _host = ReadArgument("-remote-mirror-host", "127.0.0.1");
            if (!int.TryParse(ReadArgument("-remote-mirror-port", "7001"), out _port) || _port < 1 || _port > 65535)
            {
                Fail("Invalid -remote-mirror-port");
                return;
            }
            StartCoroutine(WaitForSceneThenConnect());
        }

        private System.Collections.IEnumerator WaitForSceneThenConnect()
        {
            // Iterator methods cannot contain unsafe pointer access.  Keep the
            // readiness probe in a regular unsafe helper instead.
            while (!IsMuJoCoSceneReady())
                yield return null;

            var training = FindObjectOfType<TrainingServer>();
            _localSceneHash = training != null ? training.GetSimSceneHash() : "";
            if (string.IsNullOrEmpty(_localSceneHash))
            {
                Fail("Could not read local MuJoCo scene hash");
                yield break;
            }

            // A mirror must not run its own FixedUpdate physics between remote
            // snapshots.  Rendering still works because ApplyState explicitly
            // forwards MuJoCo and syncs the Unity component transforms.
            MjScene.Instance.enabled = false;
            var controller = FindObjectOfType<UnityRobotEnv.Core.MjJoyConController>();
            if (controller != null) controller.enabled = false;
            var joycon = FindObjectOfType<JoyConReceiver>();
            if (joycon != null) joycon.AutoStart = false;
            _mirrorEnabled = true;

            _running = true;
            _thread = new Thread(NetworkLoop) { IsBackground = true, Name = "RemoteMirrorClient" };
            _thread.Start();
            Debug.Log($"[RemoteMirror] enabled; connecting to {_host}:{_port}, scene={_localSceneHash}");
        }

        private static unsafe bool IsMuJoCoSceneReady()
            => MjScene.InstanceExists && MjScene.Instance.Model != null && MjScene.Instance.Data != null;

        private void Update()
        {
            while (_received.TryDequeue(out var message))
                HandleMessage(message);

            if (_from != null && _to != null && _mirrorEnabled)
            {
                float alpha = _blendDuration <= 1e-4f ? 1f : Mathf.Clamp01((Time.unscaledTime - _blendStarted) / _blendDuration);
                ApplyInterpolatedState(_from, _to, alpha);
            }
        }

        private void HandleMessage(WireMessage message)
        {
            if (!string.IsNullOrEmpty(message.error))
            {
                Fail("Remote error: " + message.error);
                return;
            }
            if (message.protocol != Protocol)
            {
                Fail($"Protocol mismatch: remote={message.protocol}, local={Protocol}");
                return;
            }
            if (!string.Equals(message.scene_hash, _localSceneHash, StringComparison.Ordinal))
            {
                Fail($"Scene hash mismatch: remote={message.scene_hash}, local={_localSceneHash}");
                return;
            }
            if (message.type == "episode_start")
            {
                _episode = message.episode_id;
                _lastSequence = -1;
                _from = _to = null;
                _status = $"Episode {_episode} started";
                return;
            }
            if (message.type != "snapshot") return;
            if (message.episode_id != _episode || message.seq <= _lastSequence) return;
            if (!ValidateSnapshot(message)) return;
            message = RemapToLocalLayout(message);

            _lastSequence = message.seq;
            if (_to == null)
            {
                _from = _to = message;
                ApplyInterpolatedState(message, message, 1f);
            }
            else
            {
                // Blend from the currently rendered qpos, not the previous
                // packet, so a late packet cannot cause a visible jump.
                _from = CaptureRenderedState(_to);
                _to = message;
                _blendStarted = Time.unscaledTime;
                _blendDuration = DefaultSnapshotPeriod;
            }
            _status = $"Episode {_episode} | snapshot {_lastSequence}";
        }

        private unsafe bool ValidateSnapshot(WireMessage s)
        {
            var model = MjScene.Instance.Model;
            bool valid = s.qpos != null && s.qvel != null && s.act != null && s.ctrl != null
                && s.qpos.Length == model->nq && s.qvel.Length == model->nv
                && s.act.Length == model->na && s.ctrl.Length == model->nu
                && s.joint_qposadr != null && s.joint_type != null && s.joint_names != null && s.actuator_names != null
                && s.joint_qposadr.Length == model->njnt && s.joint_type.Length == model->njnt
                && s.joint_names.Length == model->njnt && s.actuator_names.Length == model->nu;
            if (!valid) { Fail("Rejected malformed mirror snapshot"); return false; }
            if (_remoteJointToLocal == null && !BuildLayoutMaps(s)) return false;
            return _remoteJointToLocal != null && _remoteActuatorToLocal != null;
        }

        private unsafe bool BuildLayoutMaps(WireMessage remote)
        {
            var model = MjScene.Instance.Model;
            var localJoints = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal);
            for (int j = 0; j < model->njnt; j++)
            {
                string name = StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_JOINT, j);
                if (localJoints.ContainsKey(name)) { Fail("Local mirror has duplicate joint name: " + name); return false; }
                localJoints.Add(name, j);
            }
            _remoteJointToLocal = new int[remote.joint_names.Length];
            for (int r = 0; r < remote.joint_names.Length; r++)
            {
                if (!localJoints.TryGetValue(remote.joint_names[r], out int local) || remote.joint_type[r] != model->jnt_type[local])
                { Fail("Rejected mirror snapshot with incompatible joint: " + remote.joint_names[r]); _remoteJointToLocal = null; return false; }
                _remoteJointToLocal[r] = local;
            }
            var localActuators = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal);
            for (int a = 0; a < model->nu; a++)
            {
                string name = StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_ACTUATOR, a);
                if (localActuators.ContainsKey(name)) { Fail("Local mirror has duplicate actuator name: " + name); _remoteJointToLocal = null; return false; }
                localActuators.Add(name, a);
            }
            _remoteActuatorToLocal = new int[remote.actuator_names.Length];
            for (int r = 0; r < remote.actuator_names.Length; r++)
            {
                if (!localActuators.TryGetValue(remote.actuator_names[r], out int local))
                { Fail("Rejected mirror snapshot with incompatible actuator: " + remote.actuator_names[r]); _remoteJointToLocal = null; _remoteActuatorToLocal = null; return false; }
                _remoteActuatorToLocal[r] = local;
            }
            Debug.Log("[RemoteMirror] mapped remote MuJoCo names onto local joint layout");
            return true;
        }

        private unsafe WireMessage RemapToLocalLayout(WireMessage remote)
        {
            var model = MjScene.Instance.Model;
            // WireMessage is a class.  Do not assign `local = remote`: doing
            // so would replace the received source qpos before it is copied.
            var local = new WireMessage
            {
                type = remote.type,
                error = remote.error,
                protocol = remote.protocol,
                scene_hash = remote.scene_hash,
                episode_id = remote.episode_id,
                seq = remote.seq,
                sim_time = remote.sim_time,
                nq = remote.nq,
                nv = remote.nv,
                na = remote.na,
                nu = remote.nu,
                qpos = new double[model->nq],
                qvel = new double[model->nv],
                act = (double[])remote.act.Clone(), // Current SO100 model has na=0.
                ctrl = new double[model->nu],
                joint_qposadr = new int[model->njnt],
                joint_type = new int[model->njnt],
                joint_names = new string[model->njnt],
                actuator_names = new string[model->nu]
            };
            int remoteDofAdr = 0;
            for (int r = 0; r < remote.joint_names.Length; r++)
            {
                int l = _remoteJointToLocal[r];
                int qposWidth = QposWidth(remote.joint_type[r]);
                int dofWidth = DofWidth(remote.joint_type[r]);
                var destinationJoint = LocalJointById(l);
                if (destinationJoint == null) throw new InvalidOperationException("Missing local mirror joint id " + l);
                Array.Copy(remote.qpos, remote.joint_qposadr[r], local.qpos, destinationJoint.QposAddress, qposWidth);
                // The mirror's MjScene is never stepped locally.  qvel is
                // therefore visual-only; leave it zero while qpos is mapped
                // by semantic joint name.  This avoids depending on a
                // platform-specific dof-address ordering in the renderer.
                remoteDofAdr += dofWidth;
                local.joint_qposadr[l] = destinationJoint.QposAddress;
                local.joint_type[l] = model->jnt_type[l];
                local.joint_names[l] = StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_JOINT, l);
            }
            for (int r = 0; r < remote.actuator_names.Length; r++)
            {
                int l = _remoteActuatorToLocal[r];
                local.ctrl[l] = remote.ctrl[r];
                local.actuator_names[l] = StableMuJoCoName(model, MujocoLib.mjtObj.mjOBJ_ACTUATOR, l);
            }
            return local;
        }

        private static MjBaseJoint LocalJointById(int id)
        {
            foreach (var joint in UnityEngine.Object.FindObjectsOfType<MjBaseJoint>())
                if (joint.MujocoId == id) return joint;
            return null;
        }

        private static int QposWidth(int type) => type == (int)MujocoLib.mjtJoint.mjJNT_FREE ? 7 : type == (int)MujocoLib.mjtJoint.mjJNT_BALL ? 4 : 1;
        private static int DofWidth(int type) => type == (int)MujocoLib.mjtJoint.mjJNT_FREE ? 6 : type == (int)MujocoLib.mjtJoint.mjJNT_BALL ? 3 : 1;

        private static unsafe string StableMuJoCoName(MujocoLib.mjModel_* model, MujocoLib.mjtObj type, int id)
        {
            // See TrainingServer: mj_id2name's string marshalling in this
            // Tuanjie/MuJoCo package is unsafe on Windows.  MjComponents keep
            // the equivalent semantic name and runtime id without native text
            // ownership crossing the managed boundary.
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
            int split = name.LastIndexOf('_');
            if (split >= 0 && name.Length - split - 1 >= 1)
            {
                bool digits = true;
                for (int i = split + 1; i < name.Length; i++) if (!char.IsDigit(name[i])) { digits = false; break; }
                if (digits) name = name.Substring(0, split);
            }
            return name;
        }

        private unsafe WireMessage CaptureRenderedState(WireMessage template)
        {
            var model = MjScene.Instance.Model;
            var data = MjScene.Instance.Data;
            var copy = new WireMessage
            {
                qpos = new double[model->nq], qvel = template.qvel, act = template.act, ctrl = template.ctrl,
                joint_qposadr = template.joint_qposadr, joint_type = template.joint_type,
                sim_time = data->time
            };
            for (int i = 0; i < model->nq; i++) copy.qpos[i] = data->qpos[i];
            return copy;
        }

        private unsafe void ApplyInterpolatedState(WireMessage from, WireMessage to, float alpha)
        {
            var scene = MjScene.Instance;
            if (scene == null || scene.Model == null || scene.Data == null) return;
            var model = scene.Model;
            var data = scene.Data;
            var qpos = new double[model->nq];
            Array.Copy(from.qpos, qpos, qpos.Length);
            for (int j = 0; j < model->njnt; j++)
            {
                int adr = model->jnt_qposadr[j];
                int type = model->jnt_type[j];
                if (type == (int)MujocoLib.mjtJoint.mjJNT_FREE)
                {
                    for (int i = 0; i < 3; i++) qpos[adr + i] = Mathf.Lerp((float)from.qpos[adr + i], (float)to.qpos[adr + i], alpha);
                    LerpQuaternionWxyz(from.qpos, to.qpos, qpos, adr + 3, alpha);
                }
                else if (type == (int)MujocoLib.mjtJoint.mjJNT_BALL)
                {
                    LerpQuaternionWxyz(from.qpos, to.qpos, qpos, adr, alpha);
                }
                else
                {
                    qpos[adr] = Mathf.Lerp((float)from.qpos[adr], (float)to.qpos[adr], alpha);
                }
            }
            for (int i = 0; i < model->nq; i++) data->qpos[i] = qpos[i];
            for (int i = 0; i < model->nv; i++) data->qvel[i] = to.qvel[i];
            for (int i = 0; i < model->na; i++) data->act[i] = to.act[i];
            for (int i = 0; i < model->nu; i++) data->ctrl[i] = to.ctrl[i];
            data->time = Mathf.Lerp((float)from.sim_time, (float)to.sim_time, alpha);
            MujocoLib.mj_forward(model, data);
            scene.SyncUnityToMjState();
        }

        private static void LerpQuaternionWxyz(double[] a, double[] b, double[] output, int adr, float alpha)
        {
            var qa = new Quaternion((float)a[adr + 1], (float)a[adr + 2], (float)a[adr + 3], (float)a[adr]);
            var qb = new Quaternion((float)b[adr + 1], (float)b[adr + 2], (float)b[adr + 3], (float)b[adr]);
            var q = Quaternion.Slerp(qa, qb, alpha).normalized;
            output[adr] = q.w; output[adr + 1] = q.x; output[adr + 2] = q.y; output[adr + 3] = q.z;
        }

        public void RequestReset()
        {
            if (_mirrorEnabled) _outbound.Enqueue(JsonUtility.ToJson(new ResetMessage()));
        }

        private void NetworkLoop()
        {
            while (_running)
            {
                try
                {
                    using var client = new TcpClient();
                    client.NoDelay = true;
                    client.Connect(_host, _port);
                    using var stream = client.GetStream();
                    using var reader = new StreamReader(stream, Encoding.UTF8, false, 65536, true);
                    using var writer = new StreamWriter(stream, new UTF8Encoding(false), 65536, true) { AutoFlush = true };
                    writer.WriteLine(JsonUtility.ToJson(new HelloMessage { scene_hash = _localSceneHash }));
                    _connected = true;
                    _status = "Connected; waiting for remote episode";
                    while (_running && client.Connected)
                    {
                        while (_outbound.TryDequeue(out string outbound)) writer.WriteLine(outbound);
                        string line = reader.ReadLine();
                        if (line == null) break;
                        string lower = line.ToLowerInvariant();
                        if (lower.Contains("base64") || lower.Contains("agentview") || lower.Contains("eye_in_hand") || lower.Contains("png"))
                        {
                            Fail("Rejected image-bearing message on mirror channel");
                            break;
                        }
                        _received.Enqueue(JsonUtility.FromJson<WireMessage>(line));
                    }
                }
                catch (Exception ex)
                {
                    _status = "Disconnected: " + ex.Message;
                    Debug.LogWarning("[RemoteMirror] " + _status);
                }
                finally { _connected = false; }
                if (_running) Thread.Sleep(1000);
            }
        }

        private void Fail(string reason)
        {
            _status = reason;
            Debug.LogError("[RemoteMirror] " + reason);
        }

        private static string ReadArgument(string name, string fallback)
            => TryReadArgument(name, out string value) ? value : fallback;

        private static bool TryReadArgument(string name, out string value)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                { value = args[i + 1]; return true; }
            value = null;
            return false;
        }

        private void OnApplicationQuit()
        {
            _running = false;
            try { _thread?.Join(500); } catch { }
        }
    }
}
