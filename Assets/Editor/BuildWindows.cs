using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using Mujoco;

/// <summary>
/// Windows standalone build entry point (command line trigger):
///   1) verify the windowsstandalonesupport module is present
///   2) create an empty scene (DefaultGameObjects: Main Camera + Directional Light;
///      without a light the agentview renders black)
///   3) import the MJCF with the official MjImporterWithAssets (MjComponent hierarchy
///      goes into the current scene; MjScene is lazily created at runtime and collects
///      via FindObjectsOfType, so nothing has to be attached by hand)
///   4) save the scene -> BuildPipeline.BuildPlayer(StandaloneWindows64)
///
/// The scene is passed through BuildPlayerOptions.scenes, so EditorBuildSettings.asset
/// is left untouched.
///
/// Why a standalone build: the editor on this machine is fragile (a D3D11 device-reset
/// crash left it refusing to start until its stale ILPP child was killed), and it must
/// hold a licence and a Temp lock. A player runs the same bootstrap -- all the servers
/// (JoyConReceiver 5555 / TrainingServer 5556) are created by
/// [RuntimeInitializeOnLoadMethod] -- with no editor.
/// </summary>
public static class BuildWindows
{
    const string XmlAsset =
        "Assets/UnityRobotEnv/assets/robots/so100_mjcf/libero_put_block_in_box.xml";
    const string SceneAsset =
        "Assets/Local/Build/libero_put_block_in_box.build.scene";
    const string OutDir = "Build/Windows";
    const string ExeName = "unity-robot-env.exe";

    public static void Build()
    {
        string pe = Path.Combine(EditorApplication.applicationContentsPath,
            "PlaybackEngines", "windowsstandalonesupport");
        if (!Directory.Exists(pe))
            throw new System.Exception(
                "[BuildWindows] windowsstandalonesupport NOT installed: " + pe +
                " -- install 'Windows Build Support' in Tuanjie Hub and retry.");

        if (!AssetDatabase.IsValidFolder("Assets/Local/Build"))
        {
            if (!AssetDatabase.IsValidFolder("Assets/Local"))
                AssetDatabase.CreateFolder("Assets", "Local");
            AssetDatabase.CreateFolder("Assets/Local", "Build");
        }

        var scene = EditorSceneManager.NewScene(
            NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        Debug.Log("[BuildWindows] NewScene created, importing " + XmlAsset);

        // NOTE: do NOT change camera clear color / lighting here. Verified
        // 2026-09-21: the DefaultGameObjects scene renders get_obs images visually
        // identical to the old build the demos were collected with (side-by-side in
        // tools/out/old_vs_new.png). The apparent brightness difference that
        // motivated a "black background" edit was an apples-to-oranges comparison:
        // MJPEG stream frames are full-frame 224x224 camera renders (no letterbox),
        // while get_obs images are 16:9 renders letterboxed into 224x224 with black
        // bars. The model consumes the letterboxed ones, which match.

        var importer = new MjImporterWithAssets();
        var root = importer.ImportFile(XmlAsset);
        if (root == null)
            throw new System.Exception("[BuildWindows] ImportFile returned null: " + XmlAsset);
        Debug.Log("[BuildWindows] Imported root: " + root.name +
                  " at " + root.transform.position);

        if (!EditorSceneManager.SaveScene(scene, SceneAsset, true))
            throw new System.Exception("[BuildWindows] SaveScene failed: " + SceneAsset);

        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Standalone,
            ScriptingImplementation.Mono2x);

        // Incremental GC: spreads collection work across frames. The MJPEG viewer's
        // worker thread allocates ~350 KB/frame; without this, stop-the-world pauses
        // land on the control loop and measurably reduce task success at high fps.
        PlayerSettings.gcIncremental = true;

        // D3D11 first (the local GPU renders this fine in Play mode) with Vulkan as a
        // fallback, mirroring how the Linux script orders its backends.
        PlayerSettings.SetGraphicsAPIs(
            BuildTarget.StandaloneWindows64,
            new[] { GraphicsDeviceType.Direct3D11, GraphicsDeviceType.Vulkan });
        Debug.Log("[BuildWindows] GraphicsAPIs = "
                  + string.Join(", ", PlayerSettings.GetGraphicsAPIs(
                      BuildTarget.StandaloneWindows64)));

        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { SceneAsset },
            locationPathName = Path.Combine(OutDir, ExeName),
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None
        });
        var s = report.summary;
        Debug.Log($"[BuildWindows] result={s.result} errors={s.totalErrors} " +
                  $"warnings={s.totalWarnings} size={s.totalSize} out={s.outputPath} " +
                  $"time={s.totalTime.TotalMinutes:F1}min");
        if (s.totalErrors > 0 || s.result != BuildResult.Succeeded)
            throw new System.Exception("[BuildWindows] BuildPlayer failed: " + s.result);
    }
}
