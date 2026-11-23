@echo off
setlocal
rem qwen-hyperdrive Windows server launcher.
rem Pins the best measured tune (matvec 8x1 subgroup 64 + cm_gemm prefill) for this
rem process only, so plain autotune cache luck is not involved. Edit MODEL / PORT
rem below as needed. Extra arguments are passed through to the server.
set QE_VK_TUNE=8,1,64,8,1,cm_gemm_16x16x16
set MODEL=C:\Users\Yuu\.lmstudio\models\0bserverx\RVN-IQ3_S-MTP\RVN-IQ3_S-mtp.gguf
set PORT=8080
cd /d "%~dp0"
set "SERVER=%~dp0qwen35-server-vk.exe"
if not exist "%SERVER%" set "SERVER=%~dp0..\build\qwen35-server-vk.exe"
if not exist "%SERVER%" (
    echo qwen35-server-vk.exe was not found beside the launcher or in ..\build. 1>&2
    exit /b 1
)
"%SERVER%" --draft 3 -m "%MODEL%" --port %PORT% %*
set "SERVER_EXIT=%ERRORLEVEL%"
pause
exit /b %SERVER_EXIT%
