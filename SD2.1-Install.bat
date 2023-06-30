@echo off
winget install --id Git.Git -e --source winget
cls
winget install --id=Python.Python.3.10  -e
cls

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cls

goto prompt_download

:prompt_download
cls
echo -----------------------------------------------------------------------------
echo -            Would you like to download large files now? (y/n)              -
echo -                (you will download the large files later)                  -
echo -----------------------------------------------------------------------------
set /p download_prompt_answer=Enter Here:
if %download_prompt_answer% == "y" goto download_large
if %download_prompt_answer% == "n" goto download_small

:download_large
cls
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
goto install


:download_small
cls
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-base
goto install

:install
cls
pip install diffusers transformers accelerate scipy safetensors
cls
echo -----------------------------------------------------------------------------
echo -                  please launch generate.py to begin                       -
echo -----------------------------------------------------------------------------
pause
