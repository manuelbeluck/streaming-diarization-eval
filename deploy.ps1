#Requires -Version 5.0
<#
.SYNOPSIS
    Deploy src/ and config files to test server.
#>

$RemoteHost = "testuser@localhost:/data/streaming-diarization-eval"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$port = 2222
$tmp = Join-Path $env:TEMP "deploy-staging"

Write-Host "→ Deploying to $RemoteHost" -ForegroundColor Cyan

# Stage src/ without __pycache__ and *.pyc
$stageSrc = Join-Path $tmp "src"
robocopy "$projectRoot\src" $stageSrc /MIR /XD __pycache__ /XF *.pyc | Out-Null

scp -P $port -r $stageSrc $RemoteHost/
scp -P $port "$projectRoot\*.yaml" $RemoteHost/
# scp -P $port "$projectRoot\requirements.txt" $RemoteHost/
# scp -P $port "$projectRoot\README.md" $RemoteHost/

Remove-Item -Recurse -Force $tmp

Write-Host "Done" -ForegroundColor Green
