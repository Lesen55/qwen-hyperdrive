param(
    [Parameter(Mandatory)][ValidateSet('cuda', 'vulkan')][string]$Backend,
    [Parameter(Mandatory)][string]$BuildDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$CudaPath = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
$suffix = if ($Backend -eq 'vulkan') { '-vk' } else { '' }
$names = @("qwen35-server$suffix.exe", "qwen35-engine$suffix.exe")
$files = @($names | ForEach-Object { Join-Path $BuildDirectory $_ })
if ($Backend -eq 'cuda') {
    if (!$CudaPath) { throw 'CUDA_PATH is required to package the CUDA runtime.' }
    $dlls = @('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    $files += @($dlls | ForEach-Object { Join-Path $CudaPath "bin/$_" })
} else {
    $tools = @('qe_vk_test.exe', 'qe_test_cpu.exe')
    $files += @($tools | ForEach-Object { Join-Path $BuildDirectory $_ })
}
foreach ($file in $files) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing package input: $file" }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$archive = Join-Path $OutputDirectory "qwen-hyperdrive-windows-$Backend-x64.zip"
Compress-Archive -LiteralPath $files -DestinationPath $archive -CompressionLevel Fastest -Force
Write-Output $archive
