param(
    [Parameter(Mandatory = $true)]
    [string]$Name
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$SessionName = "${Timestamp}_${Name}"

$RawDir = Join-Path $ProjectRoot "data\raw\$SessionName"
$ProcessedDir = Join-Path $ProjectRoot "data\processed\$SessionName"
$ExportsDir = Join-Path $ProjectRoot "data\exports\$SessionName"
$ViewerDir = Join-Path $ProjectRoot 'third_party\orbbec\viewer\OrbbecViewer_v2.6.3_win_x64'
$ViewerExe = Join-Path $ViewerDir 'OrbbecViewer.exe'
$ViewerOutputDir = Join-Path $ViewerDir 'output'
$ViewerConfig = Join-Path $ViewerDir 'config\config.ini'

foreach ($dir in @($RawDir, $ProcessedDir, $ExportsDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if (-not (Test-Path $ViewerExe)) {
    Write-Error "Viewer not found: $ViewerExe"
    exit 1
}

if (Test-Path $ViewerConfig) {
    $ConfigLines = Get-Content $ViewerConfig
    $ConfigLines = $ConfigLines | ForEach-Object {
        if ($_ -match '^path=') {
            'path=./output'
        }
        else {
            $_
        }
    }
    $ConfigLines | Set-Content $ViewerConfig
}

if (Test-Path $ViewerOutputDir) {
    Remove-Item -LiteralPath $ViewerOutputDir -Recurse -Force
}

New-Item -ItemType Junction -Path $ViewerOutputDir -Target $RawDir | Out-Null

Write-Output "Created session:"
Write-Output "RAW       $RawDir"
Write-Output "PROCESSED $ProcessedDir"
Write-Output "EXPORTS   $ExportsDir"
Write-Output "Viewer output redirected to: $RawDir"

Start-Process -FilePath $ViewerExe -WorkingDirectory $ViewerDir
Write-Output "Started Orbbec Viewer"
