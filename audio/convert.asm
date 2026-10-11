; convert in.wav out.wav [rate=16000] [mono=1] [bits=16|32]: resample (and mix down)
; uses: audio\wav audio\resample
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

section .rdata
k_rate  db "rate", 0
k_mono  db "mono", 0
k_bits  db "bits", 0

section .bss
alignb 8
clip    resb CL_SIZE
conv    resb CL_SIZE

section .text
global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init
    ; cfg_args cuts up key=value args in place, so grab the file names first
    cmp qword [argc], 3
    jae .go
    say "usage: convert in.wav out.wav [rate=16000] [mono=1] [bits=16|32]", 13, 10
    mov ecx, 1
    call ExitProcess
.go:
    lea rax, [argv]
    mov rsi, [rax+8]
    mov rdi, [rax+16]
    call cfg_args
    mov rcx, rsi
    lea rdx, [clip]
    call wav_read
    test eax, eax
    jnz .read
    say "can't read that wav", 13, 10
    mov ecx, 1
    call ExitProcess
.read:
    lea rcx, [k_mono]
    xor edx, edx
    call cfg_int
    test rax, rax
    jz .chans
    lea rcx, [clip]
    call clip_mono
.chans:
    lea rcx, [k_rate]
    mov edx, 16000
    call cfg_int
    mov rbx, rax
    call time_now
    mov [rsp+32], rax
    lea rcx, [clip]
    mov edx, ebx
    lea r8, [conv]
    call clip_resample
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    lea rcx, [k_bits]
    mov edx, 16
    call cfg_int
    mov r8d, eax
    mov rcx, rdi
    lea rdx, [conv]
    call wav_write
    test eax, eax
    jnz .written
    say "couldn't write ", 0
    mov rcx, rdi
    call print_z
    say 13, 10
    mov ecx, 1
    call ExitProcess
.written:
    mov rcx, rsi
    call print_z
    say " -> "
    mov rcx, rdi
    call print_z
    say ": "
    mov ecx, [clip+CL_RATE]
    call print_dec
    say " -> "
    mov ecx, ebx
    call print_dec
    say " Hz, "
    mov ecx, [conv+CL_CH]
    call print_dec
    say " ch, "
    mov rcx, [conv+CL_FRAMES]
    call print_dec
    say " frames, resampled in "
    movsd xmm0, [rsp+40]
    mov edx, 4
    call print_fixed
    say " s", 13, 10
    call con_restore
    xor ecx, ecx
    call ExitProcess
