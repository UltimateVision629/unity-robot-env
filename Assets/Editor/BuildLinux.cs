using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using Mujoco;

/// <summary>
/// Linux standalone 构建入口（命令行触发）:
///   1) 校验 LinuxStandaloneSupport 模块存在
///   2) 新建空场景（DefaultGameObjects: Main Camera + Directional Light，缺光则 agentview 全黑）
///   3) 用官方 MjImporterWithAssets 导入 libero_put_block_in_box.xml（MjComponent 层级入当前场景；
///      MjScene 运行期懒创建 + FindObjectsOfType 收集，无需手动挂载）
///   4) 保存 scene → BuildPipeline.BuildPlayer(StandaloneLinux64)
/// 通过 BuildPlayerOptions.scenes 直接指定场景，不改动 EditorBuildSettings.asset。
/// </summary>
public static class BuildLinux
{
    const string XmlAsset =
        "Assets/UnityRobotEnv/assets/robots/so100_mjcf/libero_put_block_in_box.xml";
    const string SceneAsset =
        "Assets/Local/Build/libero_put_block_in_box.build.scene";
    const string OutDir = "Build/Linux";

    public static void Build()
    {
        string pe = Path.Combine(EditorApplication.applicationContentsPath,
            "PlaybackEngines", "LinuxStandaloneSupport");
        if (!Directory.Exists(pe))
            throw new System.Exception(
                "[BuildLinux] LinuxStandaloneSupport NOT installed: " + pe +
                " — 请在 Tuanjie Hub 为 2022.3.61t10 安装 'Linux 构建支持' 后重跑。");

        if (!AssetDatabase.IsValidFolder("Assets/Local/Build"))
        {
            if (!AssetDatabase.IsValidFolder("Assets/Local"))
                AssetDatabase.CreateFolder("Assets", "Local");
            AssetDatabase.CreateFolder("Assets/Local", "Build");
        }

        var scene = EditorSceneManager.NewScene(
            NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        Debug.Log("[BuildLinux] NewScene created, importing " + XmlAsset);

        // NOTE: do NOT change camera clear color / lighting here. Verified 2026-09-21
        // (tools/out/old_vs_new.png): the DefaultGameObjects scene renders get_obs
        // images visually identical to the old build the demos came from. The apparent
        // mismatch was MJPEG full-frame frames vs letterboxed get_obs images.

        var importer = new MjImporterWithAssets();
        var root = importer.ImportFile(XmlAsset);
        if (root == null)
            throw new System.Exception("[BuildLinux] ImportFile returned null: " + XmlAsset);
        Debug.Log("[BuildLinux] Imported root: " + root.name + " at " + root.transform.position);

        if (!EditorSceneManager.SaveScene(scene, SceneAsset, true))
            throw new System.Exception("[BuildLinux] SaveScene failed: " + SceneAsset);

        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone,
            ScriptingImplementation.Mono2x);

        // Incremental GC: spreads collection work across frames. The MJPEG viewer's
        // worker thread allocates ~350 KB/frame; without this, stop-the-world pauses
        // land on the control loop and measurably reduce task success at high fps.
        PlayerSettings.gcIncremental = true;

        // OpenGLCore 优先 —— 2026-09-21 在 AutoDL 的 RTX 4080 SUPER 上实测：
        //   Vulkan 优先  -> "No available video device" 后段错误 (signo:11)
        //   OpenGLCore   -> xvfb-run 下渲染成功 (agentview std 74.8,真实画面)
        // 两者都需要显示设备（xvfb-run 提供）；但无头云机上 Vulkan 这条路走不通，
        // 所以把 OpenGLCore 放第一位，Vulkan 作为真实 X11 桌面环境的兜底。
        // 详见 PROJECT_NOTES.md 的无头运行一节。
        PlayerSettings.SetGraphicsAPIs(
            BuildTarget.StandaloneLinux64,
            new[] { GraphicsDeviceType.OpenGLCore, GraphicsDeviceType.Vulkan });
        Debug.Log("[BuildLinux] GraphicsAPIs = "
                  + string.Join(", ", PlayerSettings.GetGraphicsAPIs(
                      BuildTarget.StandaloneLinux64)));

        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { SceneAsset },
            locationPathName = Path.Combine(OutDir, "unity-robot-env"),
            target = BuildTarget.StandaloneLinux64,
            options = BuildOptions.None
        });
        var s = report.summary;
        Debug.Log($"[BuildLinux] result={s.result} errors={s.totalErrors} " +
                  $"warnings={s.totalWarnings} size={s.totalSize} out={s.outputPath} " +
                  $"time={report.summary.totalTime.TotalMinutes:F1}min");
        if (s.totalErrors > 0 || s.result != BuildResult.Succeeded)
            throw new System.Exception("[BuildLinux] BuildPlayer failed: " + s.result);
    }
}
