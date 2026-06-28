@echo off
REM Background audio download script.
REM Works regardless of PC/clone path: uses this file's own location (%~dp0)
REM to find the project root. Do not hardcode an absolute path here.
REM (Comments are English-only because cmd.exe on Korean Windows reads .cmd
REM  files in CP949, and UTF-8 Korean text gets mis-tokenized into bogus
REM  "command not found" errors even inside REM lines.)
REM
REM Usage: scripts\run_download.cmd        -> default 30 TARs (current behavior)
REM        scripts\run_download.cmd 60     -> 60 TARs (or any other count)

cd /d "%~dp0.."

set MAX_TARS=%1
if "%MAX_TARS%"=="" set MAX_TARS=30

set PYTHONPATH=.
python -u scripts\download_audio.py --top-n 5 --max-tars %MAX_TARS% --parallel 3 --out data\audio --meta-out artifacts\subset_meta.csv
