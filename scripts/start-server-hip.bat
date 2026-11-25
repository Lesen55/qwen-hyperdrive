@echo off
setlocal
rem qwen-hyperdrive HIP server launcher (RX 7900 XTX).
rem Binds 0.0.0.0 so the WebUI/API is reachable from the LAN and over Tailscale
rem (owner lives alone and approved unrestricted LAN access, 2026-11-20).
rem If Tailscale access still times out, the Windows Firewall inbound rule for
rem port 8081 is the next suspect. Edit MODEL / PORT below as needed.
rem Extra arguments are passed through to the server.
set MODEL=C:\Users\Yuu\.lmstudio\models\0bserverx\RVN-IQ3_S-MTP\RVN-IQ3_S-mtp.gguf
set PORT=8081
rem R4=0: 2 rows/warp in the fallback matvec (measured +10% decode on the
rem 7900 XTX, 2026-11-20; the 4-row default is tuned for sm_86).
set QE_MMV_R4=0
rem FA_WMMA=1: WMMA flash attention for prefill (measured 746 tok/s at ~12k
rem context vs 612 for the SIMT path, 813 vs 786 at ~4k, 2026-11-26). The
rem SIMT path (QE_FA_SIMT=2) remains available as a fallback via env.
set QE_FA_WMMA=1
rem GQ_T128=1: 128-token GEMM tiles (measured 786 vs 764 tok/s prefill at ~4k
rem context, no regression at ~10k, 2026-11-22).
set QE_GQ_T128=1
cd /d %~dp0
qwen35-server-hip.exe --draft 3 --host 0.0.0.0 -m "%MODEL%" --port %PORT% %*
pause
