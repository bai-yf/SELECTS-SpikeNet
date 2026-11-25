@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo Running main.py in conda env: interface
echo Working directory: %CD%
echo.

set "ACTIVATE_BAT="
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" set "ACTIVATE_BAT=C:\ProgramData\Anaconda3\Scripts\activate.bat"
if not defined ACTIVATE_BAT if exist "C:\ProgramData\Miniconda3\Scripts\activate.bat" set "ACTIVATE_BAT=C:\ProgramData\Miniconda3\Scripts\activate.bat"

if defined ACTIVATE_BAT (
    call "%ACTIVATE_BAT%" interface
) else (
    call conda activate interface
)

if errorlevel 1 (
    echo.
    echo Failed to activate conda environment: interface
    echo Please check that Anaconda/Miniconda is installed and the env name is correct.
    echo.
    pause
    exit /b 1
)

echo Python executable:
where python
echo.

python -u "main.py"
set "ERR=%ERRORLEVEL%"
echo.
echo Process exited with code %ERR%.
pause
exit /b %ERR%
