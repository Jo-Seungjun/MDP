@echo off

setlocal

if "%1" == "/h" goto help
if "%1" == "/?" goto help

set hmex=%TEMP%\haxm_extracted
set hmexs=%hmex%\source

echo Clean target directory
if exist "%hmex%" (
    del /s /q "%hmex%"
    rmdir /s /q "%hmex%"
    )

echo Create target directory
mkdir "%hmex%"

set pwd=%~dp0
set package=%pwd%IntelHaxm.exe

set haxm_sub=Intel\Download\HAXM\1.0.1

if exist "%package%" (
    echo Extract driver package.
    "%package%" --x --f "%hmexs%"

    if %PROCESSOR_ARCHITECTURE% == AMD64 (
        set source="%hmexs%\hax64.msi"
    ) else (
        set source="%hmexs%\hax.msi"
    )
) else (
    echo.
    echo Package executable could not be found: %package%
    echo Trying to process standard extracted files.
    echo.

    if %PROCESSOR_ARCHITECTURE% == AMD64 (
        set source="%ProgramFiles(x86)%\%haxm_sub%\hax64.msi"
    ) else (
        set source="%ProgramFiles%\%haxm_sub%\hax.msi"
    )
)

set hmex32=%hmex%\32
set hmex64=%hmex%\64

if %PROCESSOR_ARCHITECTURE% == AMD64 (
    set target_dir=%hmex64%
    set instfile=%hmex64%\Intel\HAXM\HaxInst64.exe
) else (
    set target_dir=%hmex32%
    set instfile=%hmex32%\Intel\HAXM\HaxInst.exe
)

set inffile=%target_dir%\Intel\HAXM\intelhaxm.inf

if not exist %source% (
    echo File %source% could not be found. Abort.
    goto error
)

echo Extracting package file: %source%
msiexec.exe /a %source% /qn TARGETDIR="%target_dir%"

if errorlevel 1 (
    echo Failed to extract files. Abort.
    goto error
)

if not exist "%instfile%" (
    echo File %instfile% could not be found. Abort.
    goto error
)

if not "%1" == "/u" (
    echo Installing driver...
    echo "%instfile%" /i "%inffile%"
    "%instfile%" /i "%inffile%"

    if errorlevel 1 (
        echo Failed to install driver.
        goto error
    )
) else (
    echo Removing driver...
    echo "%instfile%" /u "%inffile%"
    "%instfile%" /u "%inffile%"

    if errorlevel 1 (
        echo Failed to remove driver.
        goto error
    )
)

echo Done.

:success
    endlocal
    exit /b 0

:error
    endlocal
    exit /b 1

:help
    echo.
    echo Usage:
    echo %~nx0 [/u]
    echo   /u - remove driver
    echo.
    echo Note: this script should be launched with administrator permissions.
    echo.

