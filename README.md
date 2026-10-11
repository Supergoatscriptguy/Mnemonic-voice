# Mnemonic-voice

Real-time speech to text and text to speech, written entirely in assembly: x86-64 (NASM) on the CPU and
hand-written PTX on the GPU. No C, no C runtime, no Python, no CUDA toolkit, no third-party libraries.
Just Windows' own DLLs and nvcuda.dll.

It's the follow-up to [Mnemonic](https://github.com/Supergoatscriptguy/Mnemonic), a chat LLM I built the
same way. The plan:

- speech to text that streams while you talk, trained from scratch
- push-to-talk dictation that types clean, formatted text into whatever app you're in
- text to speech that starts talking almost right away
- voice chat with Mnemonic

## Status

Just getting started. The shared library (`lib\`: console, formatting, files, memory, config, timers,
RNG, thread pool, progress bar, Ctrl+C handling) comes straight from Mnemonic.

## Building

Needs NASM and link.exe (VS Build Tools) plus the Windows SDK import libs.

```
.\build.bat test\hello
bin\hello
.\test.bat
```
