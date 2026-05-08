# 12 小时后向最新 ArchivedModels/<timestamp>/ 目录写 .exit sentinel，
# 让正在跑的训练在当前 epoch 末优雅保存退出。
# 用法：
#   Start-Process powershell -WindowStyle Hidden -ArgumentList @(
#       "-ExecutionPolicy","Bypass",
#       "-File","D:\test\StudyAI\tools\schedule_soft_exit.ps1",
#       "-ModelRoot","D:\test\StudyAI\output\005_DDPM\CartoonFace",
#       "-Hours","12"
#   )

param(
    [Parameter(Mandatory=$true)] [string] $ModelRoot,
    [double] $Hours = 12.0
)

$Sec = [int]($Hours * 3600)
Write-Host "[scheduler] sleeping $Sec seconds, then writing .exit under latest run of $ModelRoot"
Start-Sleep -Seconds $Sec

# 找最新时间戳目录
$ArchDir = Join-Path $ModelRoot "ArchivedModels"
if (-not (Test-Path $ArchDir)) {
    Write-Host "[scheduler] ArchivedModels dir not found: $ArchDir"; exit 1
}
$Latest = Get-ChildItem $ArchDir -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($null -eq $Latest) {
    Write-Host "[scheduler] no run dirs under $ArchDir"; exit 1
}

$ExitPath = Join-Path $Latest.FullName ".exit"
New-Item -ItemType File -Path $ExitPath -Force | Out-Null
Write-Host "[scheduler] wrote $ExitPath"
