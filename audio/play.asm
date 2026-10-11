; play file.wav: plays it on the default output device (wasapi, shared, event driven),
; resampled to the device's rate if it needs to be
; uses: audio\wav audio\dev audio\resample
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

section .bss
alignb 8
clip    resb CL_SIZE
conv    resb CL_SIZE
dev     resb DV_SIZE

section .text
global start
start:
    push rbx
    sub rsp, 48
    call lib_init
    cmp qword [argc], 2
    jae .go
    say "usage: play file.wav", 13, 10
    mov ecx, 1
    call ExitProcess
.go:
    lea rax, [argv]
    mov rcx, [rax+8]
    lea rdx, [clip]
    call wav_read
    test eax, eax
    jnz .read
    say "can't read that wav", 13, 10
    mov ecx, 1
    call ExitProcess
.read:
    xor ecx, ecx
    lea rdx, [dev]
    mov r8d, 50
    call dev_open
    test eax, eax
    jnz .opened
    say "the output device doesn't mix in float32, not handled yet", 13, 10
    mov ecx, 1
    call ExitProcess
.opened:
    lea rcx, [dev]
    call dev_show
    lea rbx, [clip]
    mov eax, [clip+CL_RATE]
    cmp eax, [dev+DV_RATE]
    je .play
    say "resampling "
    mov ecx, [clip+CL_RATE]
    call print_dec
    say " -> "
    mov ecx, [dev+DV_RATE]
    call print_dec
    say " Hz", 13, 10
    lea rcx, [clip]
    mov edx, [dev+DV_RATE]
    lea r8, [conv]
    call clip_resample
    lea rbx, [conv]
.play:
    call time_now
    mov [rsp+32], rax
    lea rcx, [dev]
    mov rdx, rbx
    call dev_play
    test eax, eax
    jnz .played
    say "the device stopped asking for audio", 13, 10
    mov ecx, 1
    call ExitProcess
.played:
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    say "played "
    cvtsi2sd xmm0, qword [rbx+CL_FRAMES]
    cvtsi2sd xmm1, dword [rbx+CL_RATE]
    divsd xmm0, xmm1
    mov edx, 3
    call print_fixed
    say " s of audio in "
    movsd xmm0, [rsp+40]
    mov edx, 3
    call print_fixed
    say " s", 13, 10
    lea rcx, [dev]
    call dev_close
    call con_restore
    xor ecx, ecx
    call ExitProcess
