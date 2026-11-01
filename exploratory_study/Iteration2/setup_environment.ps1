$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonLauncher) {
    throw 'Python Launcher (py.exe) was not found. Install Python 3.10 or newer, then run this script again.'
}

$pythonVersion = & $pythonLauncher.Source -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw 'Could not run Python 3. Install Python 3.10 or newer and try again.'
}

$versionParts = $pythonVersion.Trim().Split('.')
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    throw "Python 3.10 or newer is required; found Python $pythonVersion."
}

$environmentPath = Join-Path $PSScriptRoot '.venv'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
if (Test-Path -LiteralPath $environmentPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'pyvenv.cfg'))) {
        throw "$environmentPath exists but is not a Python virtual environment. Move it or rename it, then rerun this script."
    }
    Write-Output "Using existing virtual environment: $environmentPath"
}
else {
    Write-Output "Creating virtual environment with Python $pythonVersion..."
    & $pythonLauncher.Source -3 -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Virtual environment creation failed.'
    }
}

Write-Output 'Installing notebook dependencies...'
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Could not upgrade pip in the virtual environment.'
}

& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Dependency installation failed. Check your internet connection and rerun this script.'
}

Write-Output ''
Write-Output 'Setup complete.'
Write-Output "Virtual environment: $environmentPath"
Write-Output 'In VS Code, open the notebook, choose Select Kernel, and select the Python interpreter in Iteration2\.venv\Scripts\python.exe.'
