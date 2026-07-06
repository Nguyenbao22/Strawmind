@echo off
:: Chuyển bảng mã sang UTF-8 để hiển thị tiếng Việt có dấu
chcp 65001 > nul
title StrawMind Dashboard - Dừng Docker

echo =================================================================
echo        Đang dừng StrawMind Web Dashboard trên Docker...
echo =================================================================
echo.

:: Di chuyển vào thư mục chứa file bat này
cd /d "%~dp0"

:: Chạy docker-compose down
docker-compose down

if %errorlevel% equ 0 (
    echo.
    echo =================================================================
    echo  [OK] Đã dừng và gỡ bỏ Container thành công!
    echo =================================================================
) else (
    echo.
    echo =================================================================
    echo  [LỖI] Không thể dừng Docker Compose.
    echo =================================================================
)

echo.
pause
