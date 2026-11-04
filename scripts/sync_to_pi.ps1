param (
    [string]$PiHost = "fe80::33cf:4028:550:b54d%14",
    [string]$PiUser = "icnclab115"
)
Write-Host "=== [1/3] 從 GitHub 拉取最新程式碼 ===" -ForegroundColor Cyan
git pull origin main

Write-Host "=== [2/3] 透過網線將程式碼傳送至樹莓派 ===" -ForegroundColor Cyan
scp -r ./src ./tests ./CLAUDE.md "${PiUser}@[${PiHost}]:~/xiaozhao-pi-edge/"

Write-Host "=== [3/3] 執行樹莓派端 Harness 測試 ===" -ForegroundColor Cyan
ssh "${PiUser}@${PiHost}" "cd ~/xiaozhao-pi-edge && python3 tests/harness.py --full"
