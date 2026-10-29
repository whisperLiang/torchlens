[CmdletBinding()]
param(
    [switch]$Broad,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$checkRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$checkPython = (Resolve-Path -LiteralPath $Python).Path
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# Load existing compiler tools into this process only. No machine settings change.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $checkVsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $checkVsWhere) {
        $checkVsRoot = & $checkVsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($checkVsRoot) {
            $checkVcvars = Join-Path $checkVsRoot 'VC\Auxiliary\Build\vcvars64.bat'
            $checkCompilerEnvironment = & cmd.exe /d /c "call `"$checkVcvars`" >nul && set"
            if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the MSVC environment.' }
            $checkCompilerVariables = @(
                'PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'VCToolsInstallDir', 'WindowsSdkDir',
                'WindowsSDKVersion', 'VCToolsVersion', 'VCINSTALLDIR', 'VSINSTALLDIR',
                'VSCMD_ARG_TGT_ARCH', 'VSCMD_VER', 'VisualStudioVersion'
            )
            foreach ($checkLine in $checkCompilerEnvironment) {
                $checkPair = $checkLine -split '=', 2
                if ($checkPair.Count -eq 2 -and $checkPair[0] -in $checkCompilerVariables) {
                    [Environment]::SetEnvironmentVariable($checkPair[0], $checkPair[1], 'Process')
                }
            }
        }
    }
}
if ($Broad -and -not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'The broad suite needs Visual Studio C++ Build Tools for torch.compile tests.'
}
$env:PATH = (Split-Path -Path $checkPython -Parent) + ';' + $env:PATH

function Invoke-QualityCheck {
    param([string[]]$CheckArguments)
    & $checkPython @CheckArguments
    if ($LASTEXITCODE -ne 0) { throw ('Quality check failed: python ' + ($CheckArguments -join ' ')) }
}

Push-Location -LiteralPath $checkRoot
try {
    Invoke-QualityCheck @('-m', 'ruff', 'format', '--check', '.')
    Invoke-QualityCheck @('-m', 'ruff', 'check', '.')
    Invoke-QualityCheck @('-m', 'mypy', 'torchlens/')
    Invoke-QualityCheck @('-m', 'pytest', 'tests/', '-m', 'smoke', '-x', '--tb=short')
    if ($Broad) {
        Invoke-QualityCheck @('-m', 'pytest', 'tests/', '-m', 'not rare and not slow', '-x', '--tb=short')
    }
} finally {
    Pop-Location
}
