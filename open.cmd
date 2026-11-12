@echo off
rem ============================================================
rem  Label Tool - mo ung dung (bam dup de chay)
rem  Lan dau (hoac khi requirements.txt thay doi) se tu goi
rem  install.cmd de cai Python va thu vien, sau do mo ung dung.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
rem Dang chay ben trong file .zip: de install.cmd bao loi va huong dan giai nen
if not exist "%~dp0app.py" goto :install
call :pick_venv
set "VENV_PY=%VENV%\Scripts\python.exe"
set "MARK=%VENV%\requirements.da-cai.txt"

if not exist "%VENV_PY%" goto :install
if not exist "%MARK%" goto :install
fc /b "%~dp0requirements.txt" "%MARK%" >nul 2>&1 || goto :install
"%VENV_PY%" -c "import PySide6.QtWidgets, PIL, yaml" >nul 2>&1 || goto :install
goto :run

:install
title Label Tool
echo  Can cai dat truoc khi mo (chi lan dau). Vui long doi...
call "%~dp0install.cmd" /nopause
if errorlevel 1 goto :fail
call :pick_venv

:run
rem pythonw: mo ung dung khong kem cua so den
start "" "%VENV%\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0

:fail
echo.
echo  Khong mo duoc Label Tool vi cai dat chua xong. Xem loi o tren.
pause
exit /b 1

rem ---- Giong het install.cmd: .venv trong thu muc nay, tru khi duong dan qua dai va chua bat Long Path ----
:pick_venv
set "VENV=%~dp0.venv"
set "BASE=%~dp0"
set "TAIL=%BASE:~70%"
if not defined TAIL exit /b 0
"%SystemRoot%\System32\reg.exe" query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | "%SystemRoot%\System32\findstr.exe" /c:"0x1" >nul && exit /b 0
set "VENV=%LOCALAPPDATA%\LabelTool\.venv"
exit /b 0
