@echo off
setlocal
rem builds every test program and every tool, then runs the self-checking tests
pushd "%~dp0"
set BAD=0

for %%t in (hello cpuinfo fmt sys rng threads progress ctrltest wav resample inflate) do (
    call .\build.bat test\%%t >nul || (echo build failed: %%t& set BAD=1)
)
rem the tools too, so nothing rots unnoticed
for %%t in (tone wavinfo play rec convert) do (
    call .\build.bat audio\%%t >nul || (echo build failed: %%t& set BAD=1)
)
for %%t in (fetch pqinfo pqcat audinfo) do (
    call .\build.bat data\%%t >nul || (echo build failed: %%t& set BAD=1)
)
if %BAD%==1 goto done

bin\cpuinfo.exe
echo.
echo == fmt
bin\fmt.exe || set BAD=1
echo == sys
bin\sys.exe lr=1e-3 foo="a b" || set BAD=1
echo == rng
bin\rng.exe || set BAD=1
echo == threads
bin\threads.exe || set BAD=1
echo == ctrl+c
bin\ctrltest.exe || set BAD=1
echo == wav
bin\wav.exe || set BAD=1
echo == resampler
bin\resample.exe || set BAD=1
echo == inflate and gzip
bin\inflate.exe || set BAD=1

:done
echo.
if %BAD%==1 (echo SOME TESTS FAILED) else (echo all tests passed)
popd
exit /b %BAD%
