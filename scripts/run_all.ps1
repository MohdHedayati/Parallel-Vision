<#
.SYNOPSIS
  Build Parallel-Vision, run every test and benchmark, and regenerate all results/figures.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Image C:\path\photo.jpg -Reps 7

Outputs (in results\): bench.csv, mpi.csv, mpi_hybrid.csv, dispatch.csv, vs_opencv.csv,
dispatch.log, vs_opencv.log, and the PNG figures. Needs MSYS2 UCRT64 (g++, OpenCV, MS-MPI,
Ninja), CMake, Microsoft MPI runtime, and Python with matplotlib + pandas.
#>
param(
    [int]$Reps = 5,
    [string]$Image = "",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH

$mpiexec = (Get-Command mpiexec -ErrorAction SilentlyContinue).Source
if (-not $mpiexec) { $mpiexec = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" }
# a Python that is NOT the MSYS2 one (that one has no matplotlib)
$py = (Get-Command python -All | Where-Object { $_.Source -notlike "*msys64*" } | Select-Object -First 1).Source

if (-not $Image) {
    foreach ($c in @("demo\input.jpg", "C:\Windows\Web\Wallpaper\Spotlight\img14.jpg")) { if (Test-Path $c) { $Image = $c; break } }
}

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }

if (-not $SkipBuild) { Step "Build (OpenMP + MPI)"; & "$root\build.ps1" -DPV_WITH_MPI=ON }
New-Item -ItemType Directory -Force results | Out-Null

Step "Unit tests (sequential vs OpenMP, calibration round-trip)"
& build\bin\pv_tests.exe

Step "MPI correctness (bit-identical to sequential, 1/2/3/4/8 ranks)"
foreach ($n in 1,2,3,4,8) { & $mpiexec -n $n build\bin\pv_mpi_tests.exe | Select-Object -Last 1 }

Step "Hybrid correctness (MPI ranks with OpenMP inside, 4 ranks x 2 threads)"
$env:PV_HYBRID = "1"; $env:OMP_NUM_THREADS = "2"
& $mpiexec -n 4 build\bin\pv_mpi_tests.exe | Select-Object -Last 1
Remove-Item Env:PV_HYBRID; Remove-Item Env:OMP_NUM_THREADS

Step "Shared-memory benchmark: sequential vs OpenMP"
cmd /c "build\bin\pv_bench.exe $Reps > results\bench.csv"

Step "MPI benchmark: 1, 2, 4, 8 ranks"
$first = $true
foreach ($n in 1,2,4,8) {
    $lines = & $mpiexec -n $n build\bin\pv_mpi_bench.exe $Reps
    if ($first) { $lines | Out-File -Encoding ascii results\mpi.csv; $first = $false }
    else        { $lines | Select-Object -Skip 1 | Out-File -Append -Encoding ascii results\mpi.csv }
}

Step "Hybrid benchmark: 2 ranks x 4 threads, 4 ranks x 2 threads"
$env:PV_HYBRID = "1"; $first = $true
foreach ($cfg in @(@(2,4), @(4,2))) {
    $env:OMP_NUM_THREADS = "$($cfg[1])"
    $lines = & $mpiexec -n $cfg[0] build\bin\pv_mpi_bench.exe $Reps
    $lines = $lines | ForEach-Object { if ($_ -like "ranks,*") { $_ } else { "$_,$($cfg[1])" } }
    if ($first) { ($lines[0] + ",threads_per_rank"), ($lines | Select-Object -Skip 1) | Out-File -Encoding ascii results\mpi_hybrid.csv; $first = $false }
    else        { $lines | Select-Object -Skip 1 | Out-File -Append -Encoding ascii results\mpi_hybrid.csv }
}
Remove-Item Env:PV_HYBRID; Remove-Item Env:OMP_NUM_THREADS

Step "Dispatcher report: heuristic vs calibrated vs always-sequential vs always-OpenMP"
cmd /c "build\bin\pv_dispatch_report.exe 7 > results\dispatch.csv 2> results\dispatch.log"
Get-Content results\dispatch.log

Step "Comparison with OpenCV (and edge agreement on a real photo)"
if ($Image) { cmd /c "build\bin\pv_vs_opencv.exe 7 `"$Image`" > results\vs_opencv.csv 2> results\vs_opencv.log" }
else        { cmd /c "build\bin\pv_vs_opencv.exe 7 > results\vs_opencv.csv 2> results\vs_opencv.log" }
Get-Content results\vs_opencv.log

if ($Image) {
    Step "Sample edge map from the CLI"
    & build\bin\pv.exe canny $Image results\sample_canny.png --backend auto
    Copy-Item $Image results\sample_input.jpg -Force
}

Step "Figures"
& $py scripts\make_figures.py results
Write-Host "`nDone. See results\ ." -ForegroundColor Green
