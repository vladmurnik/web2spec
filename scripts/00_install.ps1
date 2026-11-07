#Requires -Version 5.1
<#
.SYNOPSIS
    One-command local setup for web2spec on Windows.

.DESCRIPTION
    Creates .venv, installs the package in editable mode, optionally installs
    llama.cpp through winget, downloads the verified GGUF models, and runs the
    doctor check.  The script never writes secrets and never sends data to a
    paid API.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/00_install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/00_install.ps1 -InstallLlama
#>
[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$SkipModels,
    [switch]$InstallLlama
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $Root
try {
    Write-Host "==> web2spec setup in $Root"

    $python = $null
    $pythonArgs = @()
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.12 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $python = $py.Source
            $pythonArgs = @("-3.12")
        }
    }
    if (-not $python) {
        $candidate = Get-Command python -ErrorAction SilentlyContinue
        if ($candidate) {
            & $candidate.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $python = $candidate.Source
            }
        }
    }
    if (-not $python) {
        throw "Python 3.12+ not found. Install it from https://www.python.org/downloads/ and rerun."
    }

    $venv = Join-Path $Root ".venv"
    if (-not (Test-Path $venv)) {
        Write-Host "==> creating .venv"
        & $python @pythonArgs -m venv $venv
    }
    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "virtual environment is incomplete: $venvPython"
    }

    Write-Host "==> installing Python package"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    $target = if ($Dev) { ".[dev]" } else { "." }
    & $venvPython -m pip install -e $target
    if ($LASTEXITCODE -ne 0) { throw "package installation failed" }

    $llama = Get-Command llama-server -ErrorAction SilentlyContinue
    if (-not $llama -and $InstallLlama) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "winget not found; install llama.cpp manually: https://github.com/ggml-org/llama.cpp"
        }
        Write-Host "==> installing llama.cpp through winget"
        & $winget.Source install --id ggml.llamacpp -e --accept-package-agreements --accept-source-agreements --silent
        $llama = Get-Command llama-server -ErrorAction SilentlyContinue
    }
    if ($llama) {
        Write-Host "==> llama-server: $($llama.Source)"
    } else {
        Write-Warning "llama-server not found. Install llama.cpp and open a new terminal, or rerun with -InstallLlama."
    }

    if (-not $SkipModels) {
        Write-Host "==> downloading verified models"
        & $venvPython -m web2spec.cli download-models --only all
        if ($LASTEXITCODE -ne 0) { throw "model download or checksum verification failed" }
    }

    Write-Host "==> checking installation"
    & $venvPython -m web2spec.cli doctor
    $doctorCode = $LASTEXITCODE

    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  .venv\Scripts\web2spec.exe collect --task `"...`" --url https://example.com"
    Write-Host "  .venv\Scripts\web2spec.exe research --task `"...`""
    Write-Host "  .venv\Scripts\web2spec.exe mcp   # stdio MCP server"
    Write-Host ""
    Write-Host "MCP config for an MCP client (use an absolute path):"
    Write-Host @"
"mcp": {
  "web2spec": {
    "type": "local",
    "command": [".venv/Scripts/python.exe", "-m", "web2spec.cli", "mcp"],
    "enabled": true
  }
}
"@

    if ($doctorCode -ne 0) {
        Write-Warning "Doctor found missing pieces. Fix them and run: .venv\Scripts\python.exe -m web2spec.cli doctor"
    }
    exit $doctorCode
}
finally {
    Pop-Location
}
