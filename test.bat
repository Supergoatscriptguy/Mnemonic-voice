@echo off
setlocal
rem builds every test program and every tool, then runs the self-checking tests
pushd "%~dp0"
set BAD=0

for %%t in (hello cpuinfo fmt sys rng threads progress ctrltest wav resample inflate flac flacenc mel gpu mx) do (
    call .\build.bat test\%%t >nul || (echo build failed: %%t& set BAD=1)
)
rem the tools too, so nothing rots unnoticed
for %%t in (tone wavinfo play rec convert) do (
    call .\build.bat audio\%%t >nul || (echo build failed: %%t& set BAD=1)
)
for %%t in (fetch pqinfo pqcat audinfo heavy shards pqshard gunzip wavshard chars) do (
    call .\build.bat data\%%t >nul || (echo build failed: %%t& set BAD=1)
)
for %%t in (gpuinfo gpubench) do (
    call .\build.bat gpu\%%t >nul || (echo build failed: %%t& set BAD=1)
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
echo == flac and md5
bin\flac.exe || set BAD=1
echo == flac encoder
bin\flacenc.exe || set BAD=1
echo == fft and log-mel
bin\mel.exe || set BAD=1
echo == gpu
bin\gpu.exe || set BAD=1
echo == mxfp8
bin\mx.exe || set BAD=1

:done
echo.
if %BAD%==1 (echo SOME TESTS FAILED) else (echo all tests passed)
popd
exit /b %BAD%
