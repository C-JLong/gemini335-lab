$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ViewerDir = Join-Path $ProjectRoot 'third_party\orbbec\viewer\OrbbecViewer_v2.6.3_win_x64'
$ViewerExe = Join-Path $ViewerDir 'OrbbecViewer.exe'
$ViewerOutputDir = Join-Path $ViewerDir 'output'
$ViewerConfig = Join-Path $ViewerDir 'config\config.ini'

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
    $outputInfo = Get-Item $ViewerOutputDir
    if ($outputInfo.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Write-Output "Viewer output target: $($outputInfo.Target)"
    }
    else {
        Write-Output "Viewer output path: $ViewerOutputDir"
    }
}

Start-Process -FilePath $ViewerExe -WorkingDirectory $ViewerDir
Write-Output "Started Orbbec Viewer from $ViewerDir"
