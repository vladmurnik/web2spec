param(
    [string]$Model = "models/gguf/web2spec-Q4_K_M.gguf",
    [int]$Port = 8080,
    [int]$Ctx = 4096,
    [int]$Ngl = 99,
    [string]$LlamaCppDir = $env:LLAMA_CPP_DIR
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $LlamaCppDir) {
    throw "Задайте LLAMA_CPP_DIR или параметр -LlamaCppDir."
}
if (-not (Test-Path $Model)) {
    throw "Модель не найдена: $Model. Выполните scripts/04_convert_gguf.ps1"
}

$Candidates = @(
    (Join-Path $LlamaCppDir "build\bin\Release\llama-server.exe"),
    (Join-Path $LlamaCppDir "build\bin\Release\llama-server"),
    (Join-Path $LlamaCppDir "build\bin\llama-server.exe"),
    (Join-Path $LlamaCppDir "llama-server.exe")
)
$Server = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Server) {
    throw "llama-server не найден. Соберите llama.cpp: cmake --build build --config Release"
}

Write-Host "llama-server: $Server" -ForegroundColor Cyan
Write-Host "Модель: $Model" -ForegroundColor Cyan
Write-Host "http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host @"
Проверка:
  curl http://127.0.0.1:$Port/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":16}'
"@

& $Server --model $Model --jinja --host 127.0.0.1 --port $Port -c $Ctx -ngl $Ngl
exit $LASTEXITCODE
