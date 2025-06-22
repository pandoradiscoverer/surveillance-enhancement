@echo off
setlocal enabledelayedexpansion
title Surveillance Enhancement System - Auto Setup

echo.
echo ================================================================
echo  SURVEILLANCE ENHANCEMENT SYSTEM - AUTO INSTALLER
echo ================================================================
echo.

REM Check if everything is already installed and configured
call :check_all_installed
if !EVERYTHING_READY! == 1 (
    echo All components already installed. Starting application...
    echo.
    goto :run_app
)

echo Starting installation process...
echo.

REM =================================================================
REM CHECK AND INSTALL MINICONDA
REM =================================================================
:check_miniconda
echo [1/4] Checking Miniconda installation...
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Miniconda found
) else (
    echo × Miniconda not found. Installing...
    call :install_miniconda
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Miniconda
        pause
        exit /b 1
    )
)

REM =================================================================
REM CHECK AND INSTALL GIT
REM =================================================================
:check_git
echo.
echo [2/4] Checking Git installation...
where git >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Git found
) else (
    echo × Git not found. Installing...
    call :install_git
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install Git
        pause
        exit /b 1
    )
)

REM =================================================================
REM SETUP CONDA ENVIRONMENT
REM =================================================================
:setup_environment
echo.
echo [3/4] Setting up Conda environment...

REM Initialize conda for this session
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" 2>nul
if %errorlevel% neq 0 call "%PROGRAMDATA%\miniconda3\Scripts\activate.bat" 2>nul

REM Check if environment exists
conda env list | findstr "surveillance-enhancement" >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Environment exists
) else (
    echo Creating environment from environment.yaml...
    if not exist "environment.yaml" (
        echo ERROR: environment.yaml not found!
        echo Make sure you're running this from the project directory.
        pause
        exit /b 1
    )
    
    conda env create -f environment.yaml
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create conda environment
        pause
        exit /b 1
    )
    echo ✓ Environment created
)

REM =================================================================
REM VERIFY INSTALLATION
REM =================================================================
:verify_installation
echo.
echo [4/4] Verifying installation...

REM Activate environment and test
call conda activate surveillance-enhancement
if %errorlevel% neq 0 (
    echo ERROR: Cannot activate environment
    pause
    exit /b 1
)

REM Test Python imports
python -c "import torch; import cv2; import flask; print('✓ Core dependencies OK')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Missing Python dependencies
    echo Trying to fix...
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install opencv-python flask pillow numpy
)

echo ✓ Installation complete
echo.

REM =================================================================
REM RUN APPLICATION
REM =================================================================
:run_app
echo Starting Surveillance Enhancement System...
echo Browser will open automatically at http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================
echo.

REM Activate environment if not already active
call conda activate surveillance-enhancement 2>nul

REM Start the application
python app.py

echo.
echo Application stopped.
pause
exit /b 0

REM =================================================================
REM FUNCTIONS
REM =================================================================

:check_all_installed
set EVERYTHING_READY=1

REM Check conda
where conda >nul 2>&1
if %errorlevel% neq 0 set EVERYTHING_READY=0

REM Check git
where git >nul 2>&1
if %errorlevel% neq 0 set EVERYTHING_READY=0

REM Check environment
if !EVERYTHING_READY! == 1 (
    call conda activate surveillance-enhancement 2>nul
    if !errorlevel! neq 0 set EVERYTHING_READY=0
)

REM Check app.py exists
if not exist "app.py" set EVERYTHING_READY=0

goto :eof

:install_miniconda
echo.
echo Downloading Miniconda installer...
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set INSTALLER=%TEMP%\miniconda_installer.exe

REM Download with PowerShell
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%INSTALLER%'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Miniconda
    exit /b 1
)

echo Installing Miniconda...
REM Silent install to default location
"%INSTALLER%" /InstallationType=JustMe /RegisterPython=1 /S /D=%USERPROFILE%\miniconda3
if %errorlevel% neq 0 (
    echo ERROR: Miniconda installation failed
    exit /b 1
)

REM Add to PATH for this session
set PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%

REM Cleanup
del "%INSTALLER%" 2>nul

echo ✓ Miniconda installed successfully
goto :eof

:install_git
echo.
echo Downloading Git installer...
set GIT_URL=https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe
set INSTALLER=%TEMP%\git_installer.exe

REM Download with PowerShell
powershell -Command "& {Invoke-WebRequest -Uri '%GIT_URL%' -OutFile '%INSTALLER%'}"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Git
    exit /b 1
)

echo Installing Git...
REM Silent install with default options
"%INSTALLER%" /VERYSILENT /NORESTART
if %errorlevel% neq 0 (
    echo ERROR: Git installation failed
    exit /b 1
)

REM Add to PATH for this session
set PATH=%PROGRAMFILES%\Git\cmd;%PATH%

REM Cleanup
del "%INSTALLER%" 2>nul

echo ✓ Git installed successfully
goto :eof