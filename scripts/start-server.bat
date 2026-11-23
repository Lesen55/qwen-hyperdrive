@echo off
setlocal
rem qwen-hyperdrive Windows server launcher.
rem Edit MODEL / PORT as needed. The server autotunes for the selected GPU.
rem To override tuning, set QE_VK_TUNE before starting the launcher.
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
