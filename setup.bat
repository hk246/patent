@echo off
chcp 65001 >nul 2>&1
title 特許クリアランス調査システム - セットアップ

echo.
echo ============================================
echo   特許クリアランス調査システム セットアップ
echo ============================================
echo.

:: ── Python 確認 ──
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Python が見つかりません。
    echo Python 3.10 以上をインストールしてください。
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/3] Python を確認しました:
python --version
echo.

:: ── venv 作成 ──
if exist "venv\Scripts\activate.bat" (
    echo [2/3] 仮想環境は既に存在します（スキップ）
) else (
    echo [2/3] 仮想環境を作成しています...
    python -m venv venv
    if errorlevel 1 (
        echo [エラー] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
    echo       作成完了: venv\
)
echo.

:: ── パッケージインストール ──
echo [3/3] 依存パッケージをインストールしています...
echo       （初回は数分かかることがあります）
echo.
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [エラー] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo ============================================
echo   セットアップ完了！
echo   start.bat でアプリを起動できます。
echo ============================================
echo.
pause
