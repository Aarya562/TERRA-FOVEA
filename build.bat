@echo off
rem Builds the C++ core on Windows. NOT TESTED in the environment this project was built in (Linux).
rem Option 1 (MinGW-w64 g++ on PATH, e.g. from MSYS2 or winlibs):
rem     build.bat
rem Option 2 (Visual Studio "x64 Native Tools" prompt):
rem     build.bat msvc
if not exist build mkdir build
if "%1"=="msvc" goto msvc
g++ -std=c++17 -O3 -march=native -ffast-math -fno-finite-math-only -fopenmp -shared -static-libgcc -static-libstdc++ core\terra_fovea.cpp -o build\terra_fovea.dll || goto fail
g++ -std=c++17 -O3 -march=native -ffast-math -fno-finite-math-only -fopenmp -static core\terra_cli.cpp core\terra_fovea.cpp -o build\terra_cli.exe || goto fail
goto ok
:msvc
cl /nologo /std:c++17 /O2 /fp:fast /openmp /EHsc /LD core\terra_fovea.cpp /Fe:build\terra_fovea.dll /Fo:build\ || goto fail
cl /nologo /std:c++17 /O2 /fp:fast /openmp /EHsc core\terra_cli.cpp core\terra_fovea.cpp /Fe:build\terra_cli.exe /Fo:build\ || goto fail
:ok
echo built build\terra_fovea.dll and build\terra_cli.exe
build\terra_cli.exe selftest
exit /b 0
:fail
echo BUILD FAILED. The dashboard still works without the C++ core: python scripts\run_dashboard.py --engine python
exit /b 1
