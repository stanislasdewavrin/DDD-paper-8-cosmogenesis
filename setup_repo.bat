@echo off
REM ====================================================================
REM  setup_repo.bat — Clean LaTeX build artefacts + initialise git repo
REM  for Paper VIII (Cosmogenesis & DESI DR2)
REM
REM  Usage:
REM    1. cd into this folder (paperVIII_cosmogenesis)
REM    2. Run: setup_repo.bat
REM    3. After it completes, push to GitHub with:
REM         git remote add origin https://github.com/stanislasdewavrin/DDD-paper-8-cosmogenesis.git
REM         git branch -M main
REM         git push -u origin main
REM ====================================================================

setlocal enableextensions
cd /d "%~dp0"

echo.
echo === Step 1/4 : Cleaning LaTeX build artefacts ===
echo.
del /q /f paper.aux paper.bbl paper.blg paper.log paper.out paper.toc paper.synctex.gz 2>nul
del /q /f paper_v*.aux paper_v*.bbl paper_v*.blg paper_v*.log paper_v*.out paper_v*.toc paper_v*.pdf 2>nul
del /q /f paper_v*_legacy.tex 2>nul

echo === Step 2/4 : Removing Python build artefacts ===
echo.
if exist code\__pycache__ rmdir /s /q code\__pycache__
if exist __pycache__ rmdir /s /q __pycache__
del /q /f code\*.pyc 2>nul

echo === Step 3/4 : Writing .gitignore ===
echo.
(
  echo # LaTeX build artefacts
  echo *.aux
  echo *.bbl
  echo *.blg
  echo *.log
  echo *.out
  echo *.toc
  echo *.synctex.gz
  echo.
  echo # Versioned drafts (we keep only paper.pdf / paper.tex^)
  echo paper_v*.pdf
  echo paper_v*.tex
  echo paper_v*.aux
  echo paper_v*.bbl
  echo paper_v*.blg
  echo paper_v*.log
  echo paper_v*.out
  echo paper_v*.toc
  echo paper_ttd/
  echo.
  echo # Python
  echo __pycache__/
  echo *.pyc
  echo .ipynb_checkpoints/
  echo *.egg-info/
  echo.
  echo # Heavy data (kept locally only - regenerate with code/^)
  echo data/*.npz
  echo data/ensemble.npz
  echo.
  echo # OS
  echo .DS_Store
  echo Thumbs.db
  echo desktop.ini
) > .gitignore

echo === Step 4/4 : Initialising git repository ===
echo.

if exist .git (
    echo .git already exists - skipping git init.
) else (
    git init
    if errorlevel 1 (
        echo.
        echo ERROR: git not found in PATH. Install Git for Windows then re-run.
        exit /b 1
    )
)

git add .
git commit -m "Paper VIII v11 - Cosmogenesis and DESI DR2 (initial public release)"

echo.
echo ====================================================================
echo  Done. Repository is ready.
echo.
echo  To push to GitHub, run:
echo    git remote add origin https://github.com/stanislasdewavrin/DDD-paper-8-cosmogenesis.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo  If the remote already exists and you only want to push the v11 update:
echo    git push
echo ====================================================================
echo.
endlocal
