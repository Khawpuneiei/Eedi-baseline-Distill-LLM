# Apply 3 queue runner (scale B, target <= 6 h of GPU time).
#
# 1. Waits for ~/.kaggle/kaggle.json, downloads the Eedi competition data, prepares it.
# 2. Waits until Apply 2 (Distill run_matrix) and Reproduce 3 (token-gated KD) have finished
#    and the GPU has been idle for 10 minutes. Create outputs\apply3_queue\FORCE_START to skip.
# 3. Trains retriever + reranker, generates teacher rationales, trains the rationale reranker,
#    evaluates all three matched conditions.
# 4. Copies the report into results\, appends the DEVLOG ledger, commits and pushes.
#
# Launch detached:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\queue_apply3.ps1'

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$queueDir = Join-Path $root 'outputs\apply3_queue'
$logDir = Join-Path $queueDir 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $queueDir 'queue.log'
$statusFile = Join-Path $queueDir 'status.txt'
$forceFile = Join-Path $queueDir 'FORCE_START'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$distillRoot = Split-Path -Parent $root
$reproduceRoot = Join-Path (Split-Path -Parent $distillRoot) 'Reproduce_Token-gated_KD'
$competition = 'eedi-mining-misconceptions-in-mathematics'
$teacher = 'Qwen/Qwen2.5-1.5B-Instruct'

function Log([string]$message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $message
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Set-Status([string]$message) {
    Set-Content -Path $statusFile -Value $message -Encoding UTF8
    Log "STATUS: $message"
}

$script:stepNumber = 0
function Step([string]$name, [string[]]$arguments) {
    $script:stepNumber++
    $tag = '{0:D2}_{1}' -f $script:stepNumber, $name
    $out = Join-Path $logDir "$tag.out.log"
    $err = Join-Path $logDir "$tag.err.log"
    Set-Status "running $tag"
    Log ("python " + ($arguments -join ' '))
    $started = Get-Date
    $proc = Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $root `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait -PassThru
    $seconds = [int]((Get-Date) - $started).TotalSeconds
    if ($proc.ExitCode -ne 0) {
        Set-Status "FAILED at $tag (exit $($proc.ExitCode)); see $err"
        exit 1
    }
    Log "done $tag in ${seconds}s"
}

function Get-Cmdlines {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe'" |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine } |
        ForEach-Object { $_.CommandLine }
}

function Test-Apply2Running {
    @(Get-Cmdlines | Where-Object {
        $_ -match 'run_rest\.ps1|scripts\.(run_matrix|train_lora|evaluate_model|build_selection|profile_student)\b'
    }).Count -gt 0
}

function Test-Reproduce3Running {
    @(Get-Cmdlines | Where-Object {
        $_ -match 'scripts\.(train_kd|run_all|calibrate_tau|evaluate|report)\b'
    }).Count -gt 0
}

function Get-GpuUsedMiB {
    try {
        $value = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
        return [int]$value
    } catch { return 99999 }
}

# ---------------------------------------------------------------- data
$raw = Join-Path $root 'data\raw'
$needed = @('train.csv', 'test.csv', 'misconception_mapping.csv')
function Test-RawData { -not ($needed | Where-Object { -not (Test-Path (Join-Path $raw $_)) }) }

Log '==== Apply 3 queue started (scale B) ===='
if (-not (Test-RawData)) {
    # Kaggle's KGAT_ API tokens live in ~/.kaggle/access_token and need kaggle>=1.8 (Python 3.11+),
    # so the CLI gets its own venv; the legacy kaggle.json also works with it.
    $kaggleDir = Join-Path $env:USERPROFILE '.kaggle'
    $python312 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    $kaggleVenv = Join-Path $root '.tools\kaggle'
    while (-not (Test-RawData)) {
        if (-not ((Test-Path (Join-Path $kaggleDir 'access_token')) -or (Test-Path (Join-Path $kaggleDir 'kaggle.json')))) {
            Set-Status "waiting for $kaggleDir\access_token or kaggle.json (or the three CSVs in data\raw)"
            Start-Sleep -Seconds 120
            continue
        }
        Set-Status 'downloading Eedi data from Kaggle'
        New-Item -ItemType Directory -Force $raw | Out-Null
        $kaggle = Join-Path $kaggleVenv 'Scripts\kaggle.exe'
        if (-not (Test-Path $kaggle)) {
            & $python312 -m venv $kaggleVenv
            $p = Start-Process -FilePath (Join-Path $kaggleVenv 'Scripts\python.exe') -ArgumentList '-m','pip','install','-q','kaggle' `
                -RedirectStandardOutput (Join-Path $logDir 'pip_kaggle.out.log') `
                -RedirectStandardError (Join-Path $logDir 'pip_kaggle.err.log') -NoNewWindow -Wait -PassThru
            if ($p.ExitCode -ne 0) { Log 'pip install kaggle failed; retrying in 10 min'; Start-Sleep 600; continue }
        }
        $p = Start-Process -FilePath $kaggle -ArgumentList 'competitions','download','-c',$competition,'-p',$raw `
            -RedirectStandardOutput (Join-Path $logDir 'kaggle.out.log') `
            -RedirectStandardError (Join-Path $logDir 'kaggle.err.log') -NoNewWindow -Wait -PassThru
        $zip = Join-Path $raw "$competition.zip"
        if ($p.ExitCode -ne 0 -or -not (Test-Path $zip)) {
            Set-Status 'Kaggle download failed (accept the competition rules on kaggle.com?); retrying in 10 min'
            Start-Sleep -Seconds 600
            continue
        }
        Expand-Archive -Path $zip -DestinationPath $raw -Force
    }
}
Log 'raw data present'

if (-not (Test-Path (Join-Path $root 'data\processed\manifest.json'))) {
    Step 'prepare_data' @('-m','scripts.prepare_data','--train','data/raw/train.csv','--test','data/raw/test.csv',
        '--mapping','data/raw/misconception_mapping.csv','--output-dir','data/processed',
        '--validation-fraction','0.2','--seed','42')
}

# Fetch the two small encoders while the GPU is still busy. The teacher is already cached.
$env:HF_HUB_OFFLINE = '0'
foreach ($model in @('BAAI/bge-small-en-v1.5', 'cross-encoder/ms-marco-MiniLM-L-6-v2', $teacher)) {
    Step 'download_model' @('-m','huggingface_hub.commands.huggingface_cli','download',$model)
}
$env:HF_HUB_OFFLINE = '1'

# ---------------------------------------------------------------- wait for the GPU
$idleMinutes = 0
while ($true) {
    if (Test-Path $forceFile) { Log 'FORCE_START found; skipping waits'; break }
    $apply2 = Test-Apply2Running
    $reproduceDone = Test-Path (Join-Path $reproduceRoot 'results\results.md')
    $reproduce3 = Test-Reproduce3Running
    $used = Get-GpuUsedMiB
    if (-not $apply2 -and $reproduceDone -and -not $reproduce3 -and $used -lt 1500) {
        $idleMinutes++
        if ($idleMinutes -ge 10) { break }
    } else {
        $idleMinutes = 0
    }
    Set-Status ("waiting: apply2_running={0} reproduce3_results={1} reproduce3_running={2} gpu_used={3}MiB idle_min={4}" -f `
        $apply2, $reproduceDone, $reproduce3, $used, $idleMinutes)
    Start-Sleep -Seconds 60
}

# ---------------------------------------------------------------- GPU pipeline
$gpuStart = Get-Date
Log '==== GPU pipeline start (budget 6 h) ===='

Step 'train_retriever' @('-m','scripts.train_retriever','--train','data/processed/train.jsonl',
    '--output-dir','models/retriever','--epochs','2','--batch-size','32','--max-length','256','--seed','42')

Step 'train_reranker' @('-m','scripts.train_reranker','--train','data/processed/train.jsonl',
    '--catalog','data/processed/catalog.json','--retriever','models/retriever',
    '--output-dir','models/reranker','--epochs','1','--batch-size','32','--seed','42')

Step 'evaluate_baseline' @('-m','scripts.evaluate','--validation','data/processed/validation.jsonl',
    '--catalog','data/processed/catalog.json','--retriever','models/retriever','--reranker','models/reranker',
    '--output-dir','outputs/validation','--batch-size','64')

foreach ($split in @('train', 'validation')) {
    Step "rationales_$split" @('-m','scripts.generate_rationales','--input',"data/processed/$split.jsonl",
        '--output',"data/cache/rationales_$split.jsonl",'--model',$teacher,'--batch-size','32','--max-new-tokens','48')
}

Step 'train_reranker_rationale' @('-m','scripts.train_reranker','--train','data/processed/train.jsonl',
    '--catalog','data/processed/catalog.json','--retriever','models/retriever',
    '--rationales','data/cache/rationales_train.jsonl',
    '--output-dir','models/reranker_rationale','--epochs','1','--batch-size','32','--seed','42')

Step 'evaluate_rationale' @('-m','scripts.evaluate','--validation','data/processed/validation.jsonl',
    '--catalog','data/processed/catalog.json','--retriever','models/retriever','--reranker','models/reranker',
    '--rationale-reranker','models/reranker_rationale','--rationales','data/cache/rationales_validation.jsonl',
    '--output-dir','outputs/validation_with_rationale','--batch-size','64')

$gpuHours = [math]::Round(((Get-Date) - $gpuStart).TotalHours, 2)
Log "==== GPU pipeline finished in $gpuHours h ===="

# ---------------------------------------------------------------- publish results
$results = Join-Path $root 'results'
New-Item -ItemType Directory -Force $results | Out-Null
Copy-Item 'outputs\validation_with_rationale\report.md' (Join-Path $results 'report.md') -Force
Copy-Item 'outputs\validation_with_rationale\metrics.json' (Join-Path $results 'metrics.json') -Force
Copy-Item 'data\processed\manifest.json' (Join-Path $results 'data_manifest.json') -Force
Get-ChildItem $logDir -Filter '*.log' | Where-Object { $_.Name -match '^\d\d_' } |
    Copy-Item -Destination (New-Item -ItemType Directory -Force (Join-Path $results 'logs')) -Force

Step 'append_devlog' @('-m','scripts.append_result_ledger','--metrics','results/metrics.json',
    '--devlog','DEVLOG.md','--gpu-hours',"$gpuHours",'--teacher',$teacher)

Set-Status 'committing and pushing results'
$ErrorActionPreference = 'Continue'  # git writes progress to stderr
& git add results DEVLOG.md
& git commit -m "Record Apply 3 scale-B validation results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" | Out-Null
& git push origin main 2>&1 | ForEach-Object { Log "git: $_" }
if ($LASTEXITCODE -ne 0) { Set-Status "DONE but git push failed; results are committed locally"; exit 1 }
Set-Status "DONE in $gpuHours GPU hours; results pushed"
