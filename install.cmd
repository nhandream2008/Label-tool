@echo off
rem ============================================================
rem  Label Tool - cai dat tu dong (bam dup de chay)
rem  1. Tim Python 3.10-3.14; neu chua co thi tai Python 3.14 tu
rem     python.org va cai cho rieng tai khoan nay (khong can admin).
rem  2. Tao moi truong ao .venv ngay trong thu muc nay. Neu duong dan
rem     thu muc qua dai (Windows gioi han 260 ky tu) thi dat .venv o
rem     %LOCALAPPDATA%\LabelTool\.venv thay vao do.
rem  3. Cai tat ca thu vien trong requirements.txt.
rem  Chay lai bao nhieu lan cung duoc.
rem ============================================================
setlocal EnableExtensions
chcp 65001 >nul
rem Python/pip luon dung UTF-8 (duong dan co dau tieng Viet khong lam pip bi loi)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
title Label Tool - Cai dat

call :check_unzipped || goto :fail

set "PY_VER=3.14.7"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
call :pick_venv
set "VENV_PY=%VENV%\Scripts\python.exe"
set "MARK=%VENV%\requirements.da-cai.txt"

echo.
echo  === Label Tool - cai dat ===
echo  Moi truong ao: %VENV%
echo.

rem ---- Moi truong ao co san nhung hong (vd. chep tu may khac) thi tao lai ----
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys" >nul 2>&1 || (
        echo  Moi truong .venv cu bi hong, dang tao lai...
        rmdir /s /q "%VENV%"
    )
)
if exist "%VENV_PY%" goto :have_venv

rem ---- Tim Python ----
call :find_python
if defined PY goto :make_venv

echo  Chua tim thay Python 3.10 - 3.14 tren may.
echo  Dang tai Python %PY_VER% tu python.org (khoang 30 MB)...
set "PY_SETUP=%TEMP%\python-%PY_VER%-amd64.exe"
curl.exe -L --fail --retry 3 -o "%PY_SETUP%" "%PY_URL%"
if errorlevel 1 (
    echo.
    echo  [LOI] Khong tai duoc Python. Hay kiem tra ket noi mang roi chay lai.
    goto :fail
)
echo  Dang cai Python %PY_VER% (chi cho tai khoan nay, mat 1-2 phut)...
"%PY_SETUP%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_doc=0
if errorlevel 1 (
    echo.
    echo  [LOI] Cai Python that bai. Hay tai va cai thu cong tu https://www.python.org/downloads/
    goto :fail
)
del "%PY_SETUP%" >nul 2>&1
call :find_python
if not defined PY (
    echo  [LOI] Da cai Python nhung khong tim thay. Hay khoi dong lai may roi chay lai tep nay.
    goto :fail
)

:make_venv
echo  Dung Python: %PY%
echo  Dang tao moi truong ao .venv ...
%PY% -m venv "%VENV%"
if errorlevel 1 (
    echo  [LOI] Khong tao duoc .venv
    goto :fail
)

:have_venv
echo.
echo  Dang cai thu vien (lan dau co the mat 5-15 phut vi PyTorch kha nang)...
echo.
"%VENV_PY%" -m pip install --upgrade pip --disable-pip-version-check -q
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo  [LOI] Cai thu vien that bai. Hay kiem tra mang roi chay lai install.cmd.
    echo        Neu loi co chu "Long Path": chuyen thu muc label-tool ra duong dan
    echo        ngan hon, vd. C:\LabelTool, roi chay lai.
    goto :fail
)
"%VENV_PY%" -c "import PySide6.QtWidgets, PIL, yaml, ultralytics, onnx" >nul 2>&1
if errorlevel 1 (
    echo  [LOI] Thu vien da cai nhung khong nap duoc. Hay chay lai install.cmd.
    goto :fail
)
copy /y "%~dp0requirements.txt" "%MARK%" >nul

echo.
echo  === Cai dat xong! Bam dup open.cmd de mo Label Tool. ===
echo.
if /i not "%~1"=="/nopause" pause
exit /b 0

:fail
echo.
if /i not "%~1"=="/nopause" pause
exit /b 1

rem ---- Chan truong hop bam dup ngay ben trong file .zip (Windows chi giai nen tam vao Temp) ----
:check_unzipped
set "NOT_UNZIPPED="
if not exist "%~dp0app.py" set "NOT_UNZIPPED=1"
if not exist "%~dp0requirements.txt" set "NOT_UNZIPPED=1"
set "HERE=%~dp0"
if /i not "%HERE:\AppData\Local\Temp\=%"=="%HERE%" set "NOT_UNZIPPED=1"
if not defined NOT_UNZIPPED exit /b 0
echo.
echo  [LOI] Dang chay tu ben trong file .zip (hoac thu muc tam):
echo        %~dp0
echo.
echo  Hay giai nen truoc: bam chuot phai file .zip - "Extract All..." (Giai nen tat ca),
echo  chon thu muc ngan, vd. C:\LabelTool, roi bam dup install.cmd trong thu muc do.
exit /b 1

rem ---- Chon noi dat .venv: trong thu muc nay, tru khi duong dan qua dai va Windows chua bat Long Path ----
:pick_venv
set "VENV=%~dp0.venv"
set "BASE=%~dp0"
set "TAIL=%BASE:~70%"
if not defined TAIL exit /b 0
"%SystemRoot%\System32\reg.exe" query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled 2>nul | "%SystemRoot%\System32\findstr.exe" /c:"0x1" >nul && exit /b 0
set "VENV=%LOCALAPPDATA%\LabelTool\.venv"
exit /b 0

rem ---- Tim Python 3.10-3.14: uu tien trinh khoi chay "py", roi den thu muc cai mac dinh, roi PATH ----
:find_python
set "PY="
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if defined PY exit /b 0
for %%V in (314 313 312 311 310) do (
    if not defined PY (
        if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe""
    )
)
if defined PY exit /b 0
rem "python" tren PATH (bo qua ban gia cua Microsoft Store: no khong chay duoc lenh)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1 && set "PY=python"
exit /b 0
