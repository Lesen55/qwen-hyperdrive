# bench-hyperdrive.ps1
# qwen35-server-vk を直接起動して統制ベンチを取る（手順の考え方は llama-rx7900xtx の BENCH と同じ）
#   - 捨て生成で温めてから本番生成2本を行い、レスポンスの timings を集計する
#   - 2本目を採用値とする。-Draft で MTP ドラフト長を変えて比較できる
#
# 使い方:
#   .\bench-hyperdrive.ps1 -ServerExe .\qwen35-server-vk.exe `
#       -ModelPath "C:\Users\Yuu\.lmstudio\models\0bserverx\RVN-IQ3_S-MTP\RVN-IQ3_S-mtp.gguf" `
#       -Draft 3 -Label draft3
#
# 注意: 既にサーバーが起動していると VRAM が足りなくなる。先にそちらを止めること。

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ServerExe,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [int]$Draft = -1,
    [int]$Port = 18099,
    [int]$WarmupTokens = 512,
    [int]$BenchTokens = 1000,
    [int]$ContextSize = 131072,
    [string]$Label = "run",
    [string]$ResultsDir = "$PSScriptRoot\..\bench-results"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $ResultsDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd-HHmmss"
$logPath = Join-Path $ResultsDir "$stamp-$Label.server.log"

$prompt = "日本の四季について、春・夏・秋・冬のそれぞれについて、気候の特徴、代表的な行事、旬の食べ物、自然の景色の変化を、季節ごとに600字以上かけて詳しく説明してください。全体で3000字以上の長文でお願いします。"

$serverArgs = @("-m", $ModelPath, "-c", $ContextSize, "--port", $Port, "--host", "127.0.0.1")
if ($Draft -ge 0) { $serverArgs += @("--draft", $Draft) }

Write-Host "starting qwen35-server-vk (log: $logPath, draft: $(if ($Draft -ge 0) { $Draft } else { 'default' }))"
$proc = Start-Process -FilePath $ServerExe -ArgumentList $serverArgs `
    -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err" -PassThru -NoNewWindow
try {
    $base = "http://127.0.0.1:$Port"
    $deadline = (Get-Date).AddMinutes(5)
    $healthy = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod "$base/health" -TimeoutSec 3
            if ($h) { $healthy = $true; break }
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $healthy) {
        if (Test-Path "$logPath.err") { Get-Content "$logPath.err" -Tail 40 }
        throw "server did not become healthy in 5 minutes"
    }

    function Invoke-Gen([int]$maxTokens, [string]$tag) {
        $body = @{
            messages    = @(@{ role = "user"; content = $prompt })
            max_tokens  = $maxTokens
            temperature = 0.8
            stream      = $false
        } | ConvertTo-Json -Depth 8
        Write-Host "generating ($tag, max_tokens=$maxTokens)..."
        $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 1200
        $t = $r.timings
        $acc = if ($t.draft_n -gt 0) { $t.draft_n_accepted / $t.draft_n } else { 0 }
        [PSCustomObject]@{
            Tag       = $tag
            Decode    = [math]::Round($t.predicted_per_second, 2)
            Prefill   = [math]::Round($t.prompt_per_second, 2)
            PromptN   = $t.prompt_n
            CacheN    = $t.cache_n
            GenN      = $t.predicted_n
            Accept    = [math]::Round($acc, 3)
            DraftN    = $t.draft_n
        }
    }

    $results = @()
    $results += Invoke-Gen $WarmupTokens "warmup (discard)"
    $results += Invoke-Gen $BenchTokens "bench 1"
    $results += Invoke-Gen $BenchTokens "bench 2 (adopt this one)"

    Write-Host ""
    $results | Format-Table -AutoSize
    Write-Host "full server log: $logPath"
    $results | ConvertTo-Json | Out-File (Join-Path $ResultsDir "$stamp-$Label.json") -Encoding utf8
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
