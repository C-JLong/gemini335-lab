param(
    [string]$Name = "session"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$SessionName = "${Timestamp}_${Name}"

$RawDir = Join-Path $ProjectRoot "data\raw\$SessionName"
$ProcessedDir = Join-Path $ProjectRoot "data\processed\$SessionName"
$ExportsDir = Join-Path $ProjectRoot "data\exports\$SessionName"

foreach ($dir in @($RawDir, $ProcessedDir, $ExportsDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Output "Created session:"
Write-Output "RAW       $RawDir"
Write-Output "PROCESSED $ProcessedDir"
Write-Output "EXPORTS   $ExportsDir"
