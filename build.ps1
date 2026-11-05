# Build with MSYS2 UCRT64 toolchain (MinGW g++, OpenMP, OpenCV, MS-MPI).
$env:PATH = "C:\msys64\ucrt64\bin;" + $env:PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release @args
if ($LASTEXITCODE -eq 0) { cmake --build build }
