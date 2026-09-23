using System;
using System.Collections.Concurrent;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityRobotEnv.Networking
{
    /// <summary>
    /// Optional viewer for a headless (or remote) player. Auto-created with
    /// TrainingServer, but does NOTHING until enabled.
    ///
    /// Cost model (the reason this exists in this shape):
    ///     disabled, or enabled with no viewer attached -> zero main-thread work
    ///     viewer attached -> ONE camera render queued + ONE async GPU readback
    ///                       request per tick; the GPU->CPU copy, PNG encode and
    ///                       network write all happen on a worker thread.
    ///
    /// That matters because the obvious implementation (render + ReadPixels +
    /// EncodeToJPG on the main thread) stalls the control loop 5-20 ms per tick, and
    /// we measured it destroying inference: same checkpoint, watching 0/2 vs not
    /// watching 3/3 (2026-09-21, AutoDL RTX 4080 SUPER).
    ///
    /// Implementation notes:
    ///   * AsyncGPUReadback moves the GPU->CPU copy off the critical path; its callback
    ///     runs on the main thread but only does one memcpy before handing the frame
    ///     to the worker queue.
    ///   * The worker encodes PNG with System.IO.Compression (thread-safe). Unity's
    ///     EncodeToJPG/EncodeToPNG are main-thread only -- exactly the cost being
    ///     avoided -- so the wire format is PNG, not JPEG.
    ///   * Browsers are unreliable for PNG multipart streams; use the companion
    ///     viewer tools/view_stream.py (PIL + tkinter, no extra installs).
    ///
    /// Endpoints (raw TcpListener -- see README for why not HttpListener):
    ///     GET /                 -> tiny HTML status page
    ///     GET /stream           -> multipart stream, agentview (PNG parts)
    ///     GET /stream?cam=wrist -> multipart stream, eye_in_hand
    ///     GET /status           -> JSON
    ///
    /// Runtime control over the existing :5556 JSON-line protocol:
    ///     {"cmd":"stream_start"}                     (optionally "fps":5,"cam":"agent")
    ///     {"cmd":"stream_stop"}
    ///     {"cmd":"stream_status"}
    /// </summary>
    public class MjpegStreamer : MonoBehaviour
    {
        [Header("Stream viewer (optional, default OFF)")]
        [Tooltip("Create the listener at all. Off = no listener, zero cost.")]
        public bool EnableStream = false;

        [Tooltip("HTTP port. Kept separate from JoyConReceiver 5555 / TrainingServer 5556.")]
        public int StreamPort = 5560;

        [Tooltip("Bind address. 0.0.0.0 lets a remote viewer reach it; 127.0.0.1 forbids remote access.")]
        public string BindAddress = "0.0.0.0";

        [Tooltip("Max capture passes per second. Work only happens while a viewer is connected.")]
        public int StreamFps = 10;

        [Tooltip("Render only the agentview camera (halves per-tick cost; eye_in_hand keeps its last frame).")]
        public bool AgentOnly = false;

        [Tooltip("Downscale factor for the stream. 2 = 112x112 from a 224x224 render. Cuts GC pressure and bandwidth ~4x.")]
        public int Downscale = 1;

        // ── state ────────────────────────────────────────────────────────────
        private TcpListener _listener;
        private Thread _acceptThread;
        private Thread _worker;
        private volatile bool _running;
        private volatile bool _workerRunning;
        private int _clients;

        private readonly object _frameLock = new object();
        private byte[] _agentPng;
        private byte[] _wristPng;

        private double _lastCaptureTime = double.NegativeInfinity;
        private double _captureInterval = 1.0 / 10.0;

        // diagnostics (surfaced via /status)
        private int _enq, _enc;
        private string _lastErr = "none";

        private RenderTexture _rtA, _rtW;
        private Camera _cam, _wristCam;
        private int _width = 224, _height = 224;

        private readonly ConcurrentQueue<FrameJob> _jobs = new ConcurrentQueue<FrameJob>();
        private const int MaxQueuedJobs = 6;

        // Buffer pools: the readback callback runs on the main thread and the PNG
        // encoder runs on the worker; allocating ~350 KB per frame at 30 fps makes
        // Mono's stop-the-world GC pause the control loop (same failure class as the
        // Python GC stalls documented in DatasetsCollector). Reuse instead.
        private readonly ConcurrentStack<byte[]> _rgbaPool = new ConcurrentStack<byte[]>();
        private readonly ConcurrentStack<byte[]> _rawPool = new ConcurrentStack<byte[]>();

        private readonly struct FrameJob
        {
            public readonly bool Wrist;
            public readonly byte[] Rgba;
            public readonly int W, H;
            public FrameJob(bool wrist, byte[] rgba, int w, int h)
            { Wrist = wrist; Rgba = rgba; W = w; H = h; }
        }

        public bool IsEnabled => _running;
        public int ClientCount => Volatile.Read(ref _clients);
        public bool HasFrame { get { lock (_frameLock) return _agentPng != null; } }
        public int Enqueued => Volatile.Read(ref _enq);
        public int Encoded => Volatile.Read(ref _enc);
        public string LastError => _lastErr;
        public int StreamPortActual { get; private set; }
        public bool WantsFrames => _running && ClientCount > 0;

        // ── lifecycle ────────────────────────────────────────────────────────
        void Start()
        {
            if (StreamFps <= 0) StreamFps = 10;
            _captureInterval = 1.0 / StreamFps;
            if (EnableStream) StartStream();
        }

        public bool StartStream()
        {
            if (_running) return true;

            int port = StreamPort;
            Exception last = null;
            for (int attempt = 0; attempt < 5; attempt++, port++)
            {
                try
                {
                    IPAddress bind = BindAddress == "0.0.0.0"
                        ? IPAddress.Any : IPAddress.Parse(BindAddress);
                    var l = new TcpListener(bind, port);
                    l.Start();
                    _listener = l;
                    StreamPortActual = port;
                    _running = true;
                    _acceptThread = new Thread(AcceptLoop)
                    {
                        IsBackground = true,
                        Name = "StreamAccept"
                    };
                    _acceptThread.Start();
                    StartWorker();
                    Debug.Log($"[Stream] viewer on http://{BindAddress}:{port}/  "
                              + "(/stream, /stream?cam=wrist, /status)  "
                              + $"fps<={StreamFps} agentOnly={AgentOnly} format=PNG");
                    return true;
                }
                catch (Exception ex) { last = ex; }
            }
            Debug.LogWarning("[Stream] could not start on ports " + StreamPort + ".."
                             + (StreamPort + 4) + ": " + last?.Message
                             + " -- streaming disabled; TrainingServer unaffected.");
            _running = false;
            return false;
        }

        public void StopStream()
        {
            _running = false;
            try { _listener?.Stop(); } catch { }
            _listener = null;
            _acceptThread = null;
            StreamPortActual = 0;
            lock (_frameLock) { _agentPng = null; _wristPng = null; }
            Debug.Log("[Stream] stopped");
        }

        void OnDestroy()
        {
            StopStream();
            StopWorker();
            ReleaseTargets();
        }

        void OnApplicationQuit()
        {
            StopStream();
            StopWorker();
            ReleaseTargets();
        }

        /// <summary>Apply runtime settings (stream_start command). Safe any time.</summary>
        public void ApplySettings(int fps, bool agentOnly)
        {
            StreamFps = Mathf.Clamp(fps, 1, 60);
            AgentOnly = agentOnly;
            _captureInterval = 1.0 / StreamFps;
            Debug.Log($"[Stream] settings: fps<={StreamFps} agentOnly={AgentOnly}");
        }

        // ── main thread: capture ─────────────────────────────────────────────
        void Update()
        {
            if (_workerRunning) DrainJobsIntoWorker();
        }

        /// <summary>
        /// Called every frame by TrainingServer. Renders and enqueues ONLY while a
        /// viewer is attached; the GPU->CPU copy is asynchronous (AsyncGPUReadback),
        /// so the main thread never stalls on it.
        /// </summary>
        public void StreamTick()
        {
            if (!_running || ClientCount == 0) return;

            double now = Time.unscaledTimeAsDouble;
            if (now - _lastCaptureTime < _captureInterval) return;
            if (!EnsureTargets()) return;
            if (_jobs.Count >= MaxQueuedJobs) return;   // worker behind: drop this tick

            _lastCaptureTime = now;

            if (_cam == null) return;
            _cam.targetTexture = _rtA;
            _cam.Render();
            _cam.targetTexture = null;
            // default format (ARGB32) -- an explicit RGB24 request from an ARGB32 RT
            // can fail hasError; the worker strips alpha anyway.
            AsyncGPUReadback.Request(_rtA, 0, req => Enqueue(req, false));

            if (!AgentOnly && _wristCam != null)
            {
                _wristCam.targetTexture = _rtW;
                _wristCam.Render();
                _wristCam.targetTexture = null;
                AsyncGPUReadback.Request(_rtW, 0, req => Enqueue(req, true));
            }
        }

        /// <summary>AsyncGPUReadback callback: one memcpy, then hand off. Main thread.</summary>
        private void Enqueue(AsyncGPUReadbackRequest req, bool wrist)
        {
            if (!_running || req.hasError)
            {
                _lastErr = req.hasError ? "readback hasError" : _lastErr;
                return;
            }
            if (_jobs.Count >= MaxQueuedJobs * 2)
            {
                _lastErr = "queue full";
                return;
            }

            try
            {
                var data = req.GetData<byte>();
                // pooled buffer: no allocation on the main thread
                if (!_rgbaPool.TryPop(out var rgba) || rgba.Length != _width * _height * 4)
                    rgba = new byte[_width * _height * 4];
                data.CopyTo(rgba);
                _jobs.Enqueue(new FrameJob(wrist, rgba, _width, _height));
                _enq++;
            }
            catch (Exception ex)
            {
                _lastErr = "enqueue: " + ex.Message;
            }
        }

        // ── worker thread: alpha-strip + PNG encode ──────────────────────────
        private void StartWorker()
        {
            _workerRunning = true;
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "StreamEncoder" };
            _worker.Start();
        }

        private void StopWorker()
        {
            _workerRunning = false;
            _worker = null;
        }

        private void DrainJobsIntoWorker()
        {
            // AsyncGPUReadback callbacks arrive on the main thread; move raw frames to
            // the encoder thread here (a cheap enqueue, no encoding on the main thread).
            while (_jobs.TryDequeue(out var job))
                _encodeQueue.Enqueue(job);
        }

        private readonly ConcurrentQueue<FrameJob> _encodeQueue =
            new ConcurrentQueue<FrameJob>();

        private void WorkerLoop()
        {
            while (_workerRunning)
            {
                if (!_encodeQueue.TryDequeue(out var job))
                {
                    Thread.Sleep(5);
                    continue;
                }
                byte[] png = PngEncodeJob(job);
                if (png == null) { _lastErr = "png null"; continue; }
                _enc++;
                lock (_frameLock)
                {
                    if (job.Wrist) _wristPng = png;
                    else _agentPng = png;
                }
            }
        }

        // ── minimal PNG encoder (RGBA in; colour type 2, filter 0, zlib) ─────
        private static readonly byte[] PngSignature =
            { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

        private static readonly uint[] CrcTable = BuildCrcTable();

        private static uint[] BuildCrcTable()
        {
            var t = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++)
                    c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                t[n] = c;
            }
            return t;
        }

        private static uint CrcRaw(uint seed, byte[] data, int offset, int count)
        {
            uint c = seed;
            for (int i = offset; i < offset + count; i++)
                c = CrcTable[(c ^ data[i]) & 0xFF] ^ (c >> 8);
            return c;
        }

        private static void WriteChunk(Stream s, string type, byte[] data)
        {
            var len = BitConverter.GetBytes(data.Length);
            Array.Reverse(len);
            s.Write(len, 0, 4);
            var tb = Encoding.ASCII.GetBytes(type);
            s.Write(tb, 0, 4);
            s.Write(data, 0, data.Length);
            uint c = CrcRaw(0xFFFFFFFFu, tb, 0, tb.Length);
            c = CrcRaw(c, data, 0, data.Length) ^ 0xFFFFFFFFu;
            var cb = BitConverter.GetBytes(c);
            Array.Reverse(cb);
            s.Write(cb, 0, 4);
        }

        /// <summary>
        /// RGBA frame -> PNG. Strips alpha, applies the configured downscale, reuses
        /// pooled buffers. Runs on the worker thread only.
        /// </summary>
        private byte[] PngEncodeJob(FrameJob job)
        {
            int w = job.W, h = job.H, srcStride = w * 4;
            int ds = Mathf.Clamp(Downscale, 1, 4);
            int ow = w / ds, oh = h / ds;

            // pooled filtered-scanline buffer (filter 0 + RGB rows)
            int rawLen = (ow * 3 + 1) * oh;
            if (!_rawPool.TryPop(out var raw) || raw.Length != rawLen)
                raw = new byte[rawLen];

            int dst = 0;
            for (int y = 0; y < oh; y++)
            {
                raw[dst++] = 0;                     // filter type 0
                int rowStart = y * ds * srcStride;
                for (int x = 0; x < ow; x++)
                {
                    int s = rowStart + x * ds * 4;
                    raw[dst++] = job.Rgba[s];
                    raw[dst++] = job.Rgba[s + 1];
                    raw[dst++] = job.Rgba[s + 2];
                }
            }
            // the RGBA buffer goes back to the pool for the next readback
            _rgbaPool.Push(job.Rgba);

            return PngEncodeRaw(raw, ow, oh);
        }

        /// <summary>Filtered RGB rows -> PNG bytes. Thread-safe (no Unity API).</summary>
        private static byte[] PngEncodeRaw(byte[] raw, int w, int h)
        {
            if (raw == null || raw.Length < (w * 3 + 1) * h) return null;

            // filtered scanlines are passed in ready-built (see PngEncodeJob)
            using var ms = new MemoryStream(raw.Length / 2 + 256);
            ms.Write(PngSignature, 0, 8);

            var ihdr = new byte[13];
            var wB = BitConverter.GetBytes(w); Array.Reverse(wB);
            var hB = BitConverter.GetBytes(h); Array.Reverse(hB);
            Buffer.BlockCopy(wB, 0, ihdr, 0, 4);
            Buffer.BlockCopy(hB, 0, ihdr, 4, 4);
            ihdr[8] = 8;   // bit depth
            ihdr[9] = 2;   // colour type: truecolour
            WriteChunk(ms, "IHDR", ihdr);

            // zlib stream: 0x78 0x9C + deflate(filtered) + adler32(filtered)
            var z = new MemoryStream();
            z.WriteByte(0x78);
            z.WriteByte(0x9C);
            using (var ds = new DeflateStream(z, System.IO.Compression.CompressionLevel.Fastest, true))
                ds.Write(raw, 0, raw.Length);
            uint a1 = 1, a2 = 0;
            for (int i = 0; i < raw.Length; i++)
            {
                a1 = (a1 + raw[i]) % 65521;
                a2 = (a2 + a1) % 65521;
            }
            var adler = BitConverter.GetBytes((a2 << 16) | a1);
            Array.Reverse(adler);
            var idat = new byte[(int)z.Length + 4];
            Buffer.BlockCopy(z.ToArray(), 0, idat, 0, (int)z.Length);
            Buffer.BlockCopy(adler, 0, idat, (int)z.Length, 4);
            WriteChunk(ms, "IDAT", idat);

            WriteChunk(ms, "IEND", new byte[0]);
            return ms.ToArray();
        }

        // ── helpers ──────────────────────────────────────────────────────────
        private bool EnsureTargets()
        {
            var oc = FindObjectOfType<Core.ObservationCollector>();
            if (oc == null) return false;

            if (_cam == null) _cam = oc.AgentviewCamera;
            if (_wristCam == null) _wristCam = oc.EyeInHandCamera;
            if (_cam == null) return false;

            int w = Mathf.Max(16, oc.ImageWidth);
            int h = Mathf.Max(16, oc.ImageHeight);
            if (_rtA == null || _width != w || _height != h)
            {
                ReleaseTargets();
                _width = w; _height = h;
                _rtA = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
                _rtW = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
            }
            return _rtA != null && _rtW != null;
        }

        private void ReleaseTargets()
        {
            if (_rtA != null) { _rtA.Release(); Destroy(_rtA); _rtA = null; }
            if (_rtW != null) { _rtW.Release(); Destroy(_rtW); _rtW = null; }
        }

        private static readonly string HtmlPage =
            "<!doctype html><html><head><meta charset='utf-8'>"
            + "<title>unity-robot-env viewer</title>"
            + "<style>body{background:#111;color:#ddd;font:13px monospace;margin:12px}"
            + "img{image-rendering:pixelated;border:1px solid #333;vertical-align:top}"
            + "figure{display:inline-block;margin:0 10px 0 0}figcaption{color:#9ab}</style>"
            + "</head><body>"
            + "<p>PNG stream: best viewed with tools/view_stream.py (browsers are "
            + "unreliable for PNG multipart). Trying anyway:</p>"
            + "<figure><figcaption>agentview</figcaption>"
            + "<img src='/stream' width='448' height='448'></figure>"
            + "<figure><figcaption>eye_in_hand</figcaption>"
            + "<img src='/stream?cam=wrist' width='448' height='448'></figure>"
            + "<p id='s'>status...</p>"
            + "<script>setInterval(async()=>{try{const r=await fetch('/status');"
            + "document.getElementById('s').textContent=await r.text();}"
            + "catch(e){}},1000)</script>"
            + "</body></html>";

        // ── accept / HTTP ────────────────────────────────────────────────────
        private void AcceptLoop()
        {
            while (_running)
            {
                TcpClient c;
                try { c = _listener.AcceptTcpClient(); }
                catch (Exception)
                {
                    if (!_running) break;
                    Thread.Sleep(100);
                    continue;
                }
                var t = new Thread(() => ServeClient(c)) { IsBackground = true };
                t.Start();
            }
        }

        private void ServeClient(TcpClient client)
        {
            try
            {
                client.NoDelay = true;
                using (var stream = client.GetStream())
                {
                    string requestLine = ReadLine(stream);
                    if (string.IsNullOrEmpty(requestLine)) return;
                    string h;
                    do { h = ReadLine(stream); } while (!string.IsNullOrEmpty(h));

                    string[] parts = requestLine.Split(' ');
                    if (parts.Length < 2) return;
                    string target = parts[1];

                    if (target.StartsWith("/stream"))
                        ServeMjpeg(stream, target.IndexOf("wrist", StringComparison.OrdinalIgnoreCase) >= 0);
                    else
                        ServeText(stream, target);
                }
            }
            catch (Exception) { }
            finally
            {
                try { client.Close(); } catch { }
            }
        }

        private static string ReadLine(Stream s)
        {
            var sb = new StringBuilder();
            int b;
            while ((b = s.ReadByte()) != -1)
            {
                if (b == '\n') break;
                if (b != '\r') sb.Append((char)b);
                if (sb.Length > 4096) break;
            }
            return sb.ToString();
        }

        private static void WriteText(Stream s, string status, string ctype, string body)
        {
            byte[] buf = Encoding.UTF8.GetBytes(body);
            string head = $"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\n"
                        + $"Content-Length: {buf.Length}\r\n"
                        + "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
            byte[] hb = Encoding.ASCII.GetBytes(head);
            s.Write(hb, 0, hb.Length);
            s.Write(buf, 0, buf.Length);
            s.Flush();
        }

        private void ServeText(Stream s, string target)
        {
            if (target.StartsWith("/status"))
            {
                string json = $"{{\"enabled\":{(_running ? "true" : "false")},"
                            + $"\"clients\":{ClientCount},\"port\":{StreamPortActual},"
                            + $"\"fps\":{StreamFps},\"agent_only\":{(AgentOnly ? "true" : "false")},"
                            + $"\"format\":\"png\",\"has_frame\":{(HasFrame ? "true" : "false")},"
                            + $"\"enqueued\":{Enqueued},\"encoded\":{Encoded},"
                            + $"\"last_error\":\"{LastError}\"}}";
                WriteText(s, "200 OK", "application/json", json);
            }
            else
            {
                WriteText(s, "200 OK", "text/html; charset=utf-8", HtmlPage);
            }
        }

        private void ServeMjpeg(Stream s, bool wrist)
        {
            const string boundary = "frameboundary";
            byte[] head = Encoding.ASCII.GetBytes(
                "HTTP/1.1 200 OK\r\n"
                + "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n"
                + "Cache-Control: no-store, no-cache, must-revalidate\r\n"
                + "Connection: close\r\n\r\n");
            s.Write(head, 0, head.Length);
            s.Flush();

            Interlocked.Increment(ref _clients);
            Debug.Log($"[Stream] viewer connected ({(wrist ? "eye_in_hand" : "agentview")}), "
                      + $"clients={ClientCount}");
            try
            {
                while (_running)
                {
                    byte[] png;
                    lock (_frameLock) { png = wrist ? _wristPng : _agentPng; }
                    if (png != null && png.Length > 0)
                    {
                        byte[] part = Encoding.ASCII.GetBytes(
                            "--" + boundary + "\r\nContent-Type: image/png\r\n"
                            + "Content-Length: " + png.Length + "\r\n\r\n");
                        s.Write(part, 0, part.Length);
                        s.Write(png, 0, png.Length);
                        s.WriteByte((byte)'\r');
                        s.WriteByte((byte)'\n');
                        s.Flush();
                    }
                    Thread.Sleep(30);
                }
            }
            catch (Exception) { }
            finally
            {
                Interlocked.Decrement(ref _clients);
                Debug.Log($"[Stream] viewer disconnected, clients={ClientCount}");
            }
        }
    }
}
