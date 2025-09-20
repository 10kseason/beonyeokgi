@echo off
setlocal
REM 시스템 Python 사용 (CUDA 작동)
python -m src.main --compute-mode cuda
endlocal

