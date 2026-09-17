param(
    [ValidatePattern('^(?:[01]\d|2[0-3]):[0-5]\d$')][string]$At = '17:00',
    [ValidateSet('deepl', 'google', 'qwen')][string]$Provider = 'deepl',
    [string]$TaskName = 'KamiWiki-LocalData-Daily',
    [switch]$Preview,
    [switch]$Remove
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot 'run_data_update.ps1'
$PythonPath = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$Description = "KamiWiki local data worker: $ProjectRoot"
$Arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Runner`" -Provider $Provider"
$LegacyTaskName = 'KamiWiki-LocalData-Hourly'
if ($Preview) {
    [pscustomobject]@{ TaskName = $TaskName; Schedule = 'Daily'; At = $At;
        Command = 'powershell.exe'; Arguments = $Arguments; WorkingDirectory = $ProjectRoot;
        CatchUp = $true;
        Catalogs = @('kamihime', 'eidolon', 'weapon');
        MultipleInstances = 'IgnoreNew'; BuildsRagIndexOnce = $true; IndexDevice = 'cuda' } | ConvertTo-Json
    return
}
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing -and $Existing.Description -ne $Description) {
    throw 'Task name belongs to another workspace. Choose a different TaskName.'
}
if ($Remove) {
    if ($Existing) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }
    if ($TaskName -eq 'KamiWiki-LocalData-Daily') {
        $Legacy = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
        if ($Legacy -and $Legacy.Description -eq $Description) {
            Unregister-ScheduledTask -TaskName $LegacyTaskName -Confirm:$false
        }
    }
    return
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Run uv sync before registering the task.' }
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $Arguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)
# Current user's interactive token can access their .env and normal network credentials.
# The task runs while the user is logged in (including a locked session); no password is saved.
$Account = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $Account -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Description $Description -Action $Action `
    -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
if ($TaskName -eq 'KamiWiki-LocalData-Daily') {
    $Legacy = Get-ScheduledTask -TaskName $LegacyTaskName -ErrorAction SilentlyContinue
    if ($Legacy -and $Legacy.Description -eq $Description) {
        Unregister-ScheduledTask -TaskName $LegacyTaskName -Confirm:$false
    }
}
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State, Description
