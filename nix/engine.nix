{
  lib,
  cudaPackages,
  cmake,
  ninja,
  autoAddDriverRunpath,
  writeShellScriptBin,
  symlinkJoin,
}:

let
  core = cudaPackages.backendStdenv.mkDerivation {
    pname = "qwen35-engine";
    version = "0.1.0";

    src = lib.cleanSourceWith {
      src = ../.;
      filter =
        name: type:
        let
          rel = lib.removePrefix (toString ../. + "/") (toString name);
        in
        rel == "CMakeLists.txt" || lib.hasPrefix "src" rel || lib.hasPrefix "tools" rel || lib.hasPrefix "web" rel;
    };

    nativeBuildInputs = [
      cmake
      ninja
      cudaPackages.cuda_nvcc
      autoAddDriverRunpath
    ];

    buildInputs = [
      cudaPackages.cuda_cudart
      cudaPackages.cccl
      cudaPackages.libcublas
    ];

    cmakeFlags = [
      (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
      (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" "86-real")
    ];

    # install the test / benchmark tools too
    postInstall = ''
      for t in qe_test_kernels qe_test_cpu qe_bench_attn qe_tokenize qe_dump; do
        [ -x "$t" ] && install -Dm755 "$t" "$out/bin/$t"
      done
    '';

    meta = {
      description = "From-scratch hybrid CPU/GPU inference engine for qwen35 (Qwen3.8-27B) GGUF models";
      mainProgram = "qwen35-server";
      platforms = [ "x86_64-linux" ];
    };
  };

  # `llm-engine [server args]`: qwen35-server with the model defaulting to the Hugging Face cache copy
  launcher = writeShellScriptBin "llm-engine" ''
    # Free VRAM held by the hazkey IME server (systemd user unit) while the engine runs, and
    # restore it on exit.  Disable with LLM_ENGINE_STOP_HAZKEY=0.
    restore=0
    if [ "''${LLM_ENGINE_STOP_HAZKEY:-1}" = 1 ] && systemctl --user is-active --quiet hazkey-server.service 2>/dev/null; then
      if systemctl --user stop hazkey-server.service; then
        restore=1
        echo "llm-engine: stopped hazkey-server.service (will be restarted on exit)" >&2
        sleep 1
      fi
    fi
    cleanup() {
      if [ "$restore" = 1 ]; then
        systemctl --user start hazkey-server.service && echo "llm-engine: restarted hazkey-server.service" >&2
        restore=0
      fi
    }
    trap cleanup EXIT

    model="''${LLM_ENGINE_MODEL:-}"
    if [ -z "$model" ]; then
      for f in "$HOME"/.cache/huggingface/hub/models--huihui-ai--Huihui-Qwen3.8-27B-abliterated-GGUF/snapshots/*/*IQ3_XXS*mtp*.gguf; do
        [ -e "$f" ] && model="$f" && break
      done
    fi
    args=("$@")
    case " $* " in
      *" -m "*|*" --model "*) ;;
      *)
        if [ -z "$model" ]; then
          echo "llm-engine: no model found; pass -m <file.gguf> or set LLM_ENGINE_MODEL" >&2
          exit 1
        fi
        args=(-m "$model" "$@") ;;
    esac
    ${core}/bin/qwen35-server "''${args[@]}" &
    pid=$!
    trap 'kill -TERM $pid 2>/dev/null' INT TERM
    wait $pid
    status=$?
    wait $pid 2>/dev/null
    exit $status
  '';
in
symlinkJoin {
  name = "qwen35-engine-0.1.0";
  paths = [
    core
    launcher
  ];
  meta = core.meta // {
    mainProgram = "llm-engine";
  };
}
