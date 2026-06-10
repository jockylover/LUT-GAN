$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root "lut\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found at lut\\Scripts\\python.exe"
}

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stage1Dir = Join-Path $root "checkpoints\stage1_lut"
$stage2Dir = Join-Path $root "checkpoints\stage2_gan"
$stage1Log = Join-Path $logDir "stage1_lut.log"
$stage2Log = Join-Path $logDir "stage2_gan.log"
$evalLog = Join-Path $logDir "eval_long.log"

Write-Host "Stage 1: LUT-supervised pretrain (no GAN)"
$oldEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $python train.py `
        --content_dir data\base_content `
        --style_root data\lut_multi\style_images `
        --target_root data\lut_multi\targets `
        --lut_root style_luts `
        --save_dir $stage1Dir `
        --epochs 80 `
        --batch_size 1 `
        --lut_size 33 `
        --lambda_adv 0 `
        --lambda_lut 10 `
        --lambda_id 0.05 `
        --lambda_tv 0.05 `
        --delta_scale 0.2 `
        --no_zero_mean `
        --num_workers 0 `
        --progress_update 20 `
        --resume checkpoints\multi_lut_epoch_40.pth 2>&1 | Tee-Object -FilePath $stage1Log
} finally {
    $ErrorActionPreference = $oldEA
}
if ($LASTEXITCODE -ne 0) {
    throw "Stage 1 failed with exit code $LASTEXITCODE"
}

Write-Host "Stage 2: GAN fine-tune"
$stage1Ckpt = Join-Path $stage1Dir "lutgan_epoch_80.pth"
if (-not (Test-Path $stage1Ckpt)) {
    throw "Missing stage1 checkpoint: $stage1Ckpt"
}
$oldEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $python train.py `
        --content_dir data\base_content `
        --style_root data\lut_multi\style_images `
        --target_root data\lut_multi\targets `
        --lut_root style_luts `
        --save_dir $stage2Dir `
        --epochs 20 `
        --batch_size 1 `
        --lut_size 33 `
        --lambda_adv 0.5 `
        --lambda_lut 10 `
        --lambda_id 0.05 `
        --lambda_tv 0.05 `
        --delta_scale 0.2 `
        --no_zero_mean `
        --num_workers 0 `
        --progress_update 20 `
        --resume $stage1Ckpt 2>&1 | Tee-Object -FilePath $stage2Log
} finally {
    $ErrorActionPreference = $oldEA
}
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 failed with exit code $LASTEXITCODE"
}

Write-Host "Eval: Candlelight + Kodak 2395"
$stage2Ckpt = Join-Path $stage2Dir "lutgan_epoch_20.pth"
$oldEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $python eval_lut.py `
        --ckpt $stage2Ckpt `
        --style data\lut_multi\style_images\Candlelight\a0001-jmac_DSC1459.png `
        --lut style_luts\Candlelight.CUBE `
        --content data\base_content\a0001-jmac_DSC1459.png `
        --output_dir outputs\eval_long_candlelight 2>&1 | Tee-Object -FilePath $evalLog
} finally {
    $ErrorActionPreference = $oldEA
}
if ($LASTEXITCODE -ne 0) {
    throw "Eval (Candlelight) failed with exit code $LASTEXITCODE"
}

$oldEA = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $python eval_lut.py `
        --ckpt $stage2Ckpt `
        --style "data\lut_multi\style_images\Kodak 5218 Kodak 2395 (by Adobe)\a0001-jmac_DSC1459.png" `
        --lut "style_luts\Kodak 5218 Kodak 2395 (by Adobe).cube" `
        --content data\base_content\a0001-jmac_DSC1459.png `
        --output_dir outputs\eval_long_kodak2395 2>&1 | Tee-Object -FilePath $evalLog -Append
} finally {
    $ErrorActionPreference = $oldEA
}
if ($LASTEXITCODE -ne 0) {
    throw "Eval (Kodak 2395) failed with exit code $LASTEXITCODE"
}

Write-Host "Done."
