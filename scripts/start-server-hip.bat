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
cd /d %~dp0
qwen35-server-hip.exe --draft 3 --host 0.0.0.0 -m "%MODEL%" --port %PORT% %*
pause
