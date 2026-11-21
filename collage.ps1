$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# ---------------- CONFIG ----------------
$PythonVersion = "3.9.12"
$VenvDir = "collage_env"
$ReqFile = "requirements.txt"
$MainFile = "ui.py"
$InstallerFile = "python-installer.exe"
$LogFile = "install.log"
$FailedStep = "NONE"
# ---------------------------------------

function Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp  $msg" | Tee-Object -FilePath $LogFile -Append
}

function Rollback {
    Log "Rolling back changes"

    if (Test-Path $VenvDir) {
        Log "Removing virtual environment"
        Remove-Item $VenvDir -Recurse -Force
    }

    if (Test-Path $InstallerFile) {
        Remove-Item $InstallerFile -Force
    }
}

try {
    Log "Installation started"

    # ---------- Check Python ----------
    $FailedStep = "Checking Python"
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Log "Python not found. Downloading installer"

        $FailedStep = "Downloading Python"
        $PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
        Invoke-WebRequest $PythonUrl -OutFile $InstallerFile

        $FailedStep = "Installing Python"
        Start-Process $InstallerFile `
            -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" `
            -Wait -NoNewWindow
    } else {
        Log "Python already installed"
    }

    # ---------- Virtual Environment ----------
    if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
        $FailedStep = "Creating virtual environment"
        python -m venv $VenvDir

        $FailedStep = "Upgrading pip"
        & "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip

        $FailedStep = "Installing requirements"
        & "$VenvDir\Scripts\pip.exe" install -r $ReqFile
    } else {
        Log "Virtual environment already exists"
    }

    # ---------- Fix Qt platform plugin path ----------
    $QtPlatformPath = Join-Path `
        (Resolve-Path "$VenvDir\Lib\site-packages\PyQt6\Qt6\plugins\platforms") ""

    if (-not (Test-Path $QtPlatformPath)) {
        throw "Qt platform plugins not found at $QtPlatformPath"
    }

    [Environment]::SetEnvironmentVariable(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        $QtPlatformPath,
        "Process"
    )

    Log "QT_QPA_PLATFORM_PLUGIN_PATH set to $QtPlatformPath"


    # ---------- Run Application ----------
    $FailedStep = "Running application"

    $process = Start-Process `
        -FilePath (Resolve-Path "$VenvDir\Scripts\python.exe").Path `
        -ArgumentList "-X faulthandler `"$MainFile`"" `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardError "application-error.log" `
        -RedirectStandardOutput "application-output.log" `
        -Wait `
        -PassThru `
        -NoNewWindow
    if ($process.ExitCode -ne 0) {
        throw "Application exited with code $($process.ExitCode). See application-error.log."
    }
    exit 0
}
catch {
    Log "ERROR occurred at step: $FailedStep"
    Log $_.Exception.Message

    Write-Host ""
    Write-Host "ERROR OCCURRED" -ForegroundColor Red
    Write-Host "Failed step: $FailedStep"
    Write-Host "Message: $($_.Exception.Message)"
    Write-Host "See '$LogFile' for details"
    Write-Host ""

    exit 1
}
