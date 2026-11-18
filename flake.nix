{
  description = "llm-engine: from-scratch long-context (128K+) inference engine for Qwen3.8-27B (qwen35 hybrid DeltaNet/attention, IQ3_XXS + MTP) on RTX 3080 10GB + Ryzen 5600X";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Only used as a *reference implementation* for correctness / speed comparison (packages.reference).
    llama-cpp-src = {
      url = "github:ggml-org/llama.cpp";
      flake = false;
    };
  };

  outputs =
    { self, nixpkgs, llama-cpp-src }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          # Ampere and newer (int8 mma.sync m16n8k32 and cp.async need sm_80+)
          cudaCapabilities = [ "8.0" "8.6" "8.9" "9.0" ];
          cudaForwardCompat = false;
        };
      };
      lib = pkgs.lib;
      cuda = pkgs.cudaPackages;

      engine = pkgs.callPackage ./nix/engine.nix { cudaPackages = cuda; };
      # Vulkan only: no CUDA in the closure (AMD Radeon, Intel Arc, ...)
      engine-vulkan = pkgs.callPackage ./nix/engine.nix {
        cudaPackages = cuda;
        withCuda = false;
      };

      reference =
        (pkgs.callPackage "${llama-cpp-src}/.devops/nix/package.nix" {
          llamaVersion = "git-${llama-cpp-src.shortRev or "dirty"}";
          useCuda = true;
          useWebUi = false;
        }).overrideAttrs
          (old: {
            pname = "llama-cpp-reference";
            webui = null;
            postPatch = "";
            cmakeFlags = (old.cmakeFlags or [ ]) ++ [
              (lib.cmakeBool "GGML_AVX" true)
              (lib.cmakeBool "GGML_AVX2" true)
              (lib.cmakeBool "GGML_FMA" true)
              (lib.cmakeBool "GGML_F16C" true)
              (lib.cmakeBool "GGML_BMI2" true)
              (lib.cmakeBool "LLAMA_BUILD_TESTS" false)
            ];
          });
    in
    {
      packages.${system} = {
        default = engine;
        vulkan = engine-vulkan;
        inherit engine reference;
      };

      apps.${system} = {
        # OpenAI-compatible server (model auto-detected in the HF cache unless -m is given)
        default = {
          type = "app";
          program = "${engine}/bin/llm-engine";
        };
        server = {
          type = "app";
          program = "${engine}/bin/qwen35-server";
        };
        # CLI: perplexity / generation over token-id files (benchmarks, validation)
        cli = {
          type = "app";
          program = "${engine}/bin/qwen35-engine";
        };
      };

      devShells.${system} = {
      vulkan = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.shaderc
          pkgs.vulkan-headers
          pkgs.vulkan-loader
          pkgs.vulkan-tools
          (pkgs.python3.withPackages (ps: [ ps.numpy ps.gguf ]))
        ];
      };
      default = (pkgs.mkShell.override { stdenv = cuda.backendStdenv; }) {
        packages = [
          cuda.cuda_nvcc
          cuda.cuda_cudart
          cuda.cccl
          cuda.libcublas
          pkgs.cmake
          pkgs.ninja
          (pkgs.python3.withPackages (ps: [ ps.numpy ps.gguf ]))
          pkgs.gdb
          pkgs.vulkan-headers
          pkgs.vulkan-loader
          pkgs.shaderc
          pkgs.vulkan-tools
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          export CUDA_PATH=${cuda.cuda_nvcc}
        '';
      };
      };
    };
}
