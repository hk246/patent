@echo off
chcp 65001 >nul 2>&1
title 特許クリアランス調査システム

:: ── venv 存在チェック ──
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [エラー] 仮想環境が見つかりません。
    echo 先に setup.bat を実行してください。
    echo.
    pause
    exit /b 1
)

:: ── 起動 ──
echo.
echo   特許クリアランス調査システムを起動しています...
echo   （ブラウザが自動で開きます）
echo   終了するには このウィンドウを閉じるか Ctrl+C を押してください
echo.

call venv\Scripts\activate.bat
python flask_app.py
