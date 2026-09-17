param(
    [ValidateSet('deepl', 'google', 'qwen')]
    [string]$Provider = 'deepl',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run uv sync before starting automatic updates.' }
$RunnerPath = Join-Path $PSScriptRoot 'update_data.py'
$Arguments = @(
    $RunnerPath,
    '--object-type', 'kamihime',
    '--object-type', 'eidolon',
    '--object-type', 'weapon',
    '--provider', $Provider,
    '--scheduled'
)
if ($DryRun) { $Arguments += '--dry-run' }
Push-Location -LiteralPath $ProjectRoot
try {
    if ($DryRun) {
        & $PythonPath @Arguments
        $ResultCode = $LASTEXITCODE
    }
    else {
        $LogDir = Join-Path $ProjectRoot 'kami/data/.pipeline'
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        # Latest run output is bounded; durable per-run summaries are written by the worker.
        # Windows PowerShell wraps native stderr (including model progress) in
        # error records. Do not terminate a healthy worker because it logs there.
        $ErrorActionPreference = 'Continue'
        try {
            & $PythonPath @Arguments *> (Join-Path $LogDir 'scheduler.log')
            $ResultCode = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = 'Stop' }
    }
}
finally { Pop-Location }
exit $ResultCode
