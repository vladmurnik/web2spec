param(
    [string]$ModelDir = "models/merged/qwen2.5-0.5b-ft",
    [string]$OutDir = "models/gguf",
    [string]$LlamaCppDir = $env:LLAMA_CPP_DIR,
    [string]$Quant = "Q4_K_M",
    [string]$Name = "web2spec"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $LlamaCppDir) {
    throw "Задайте LLAMA_CPP_DIR (путь к клонированному llama.cpp) или параметр -LlamaCppDir."
}
if (-not (Test-Path $LlamaCppDir)) {
    throw "Каталог llama.cpp не найден: $LlamaCppDir"
}
if (-not (Test-Path (Join-Path $ModelDir "config.json"))) {
    throw "Нет модели: $ModelDir. Сначала выполните scripts/03_merge.py"
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"

Write-Host "Конвертация $ModelDir -> GGUF..." -ForegroundColor Cyan
& $Python -c "from pathlib import Path; from web2spec.gguf import convert_and_quantize; p = convert_and_quantize(Path(r'$ModelDir'), Path(r'$OutDir'), llama_cpp=Path(r'$LlamaCppDir'), quant_type='$Quant', name='$Name'); print(p)"
if ($LASTEXITCODE -ne 0) {
    throw "Конвертация или квантование завершились с ошибкой (код $LASTEXITCODE)"
}

$Result = Join-Path $OutDir "${Name}-$Quant.gguf"
if (-not (Test-Path $Result)) {
    throw "Ожидаемый файл не создан: $Result"
}
$SizeMb = [math]::Round((Get-Item $Result).Length / 1MB, 1)
Write-Host "Готово: $Result ($SizeMb MB)" -ForegroundColor Green
Write-Host "Запуск: scripts/06_serve.ps1 -Model `"$Result`""
