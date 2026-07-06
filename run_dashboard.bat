@echo off
:: Chuyển bảng mã sang UTF-8 để hiển thị tiếng Việt có dấu
chcp 65001 > nul
title StrawMind Dashboard - Chạy trên Docker

echo =================================================================
echo        Đang khởi chạy StrawMind Web Dashboard trên Docker...
echo =================================================================
echo.

:: Di chuyển vào thư mục chứa file bat này
cd /d "%~dp0"

:: Chạy docker-compose
docker-compose up --build -d

if %errorlevel% equ 0 (
    echo.
    echo =================================================================
    echo  [OK] Khởi chạy thành công!
    echo  Hãy mở trình duyệt và truy cập: http://localhost:5000
    echo =================================================================
) else (
    echo.
    echo =================================================================
    echo  [LỖI] Không thể khởi chạy Docker Compose. 
    echo  Vui lòng kiểm tra xem Docker Desktop đã được mở chưa!
    echo =================================================================
)

echo.
pause
