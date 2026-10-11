; wavinfo file.wav [more.wav ...]: format, length, peak and rms level of each
; uses: audio\wav
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

section .rdata
align 8
c_db    dq 8.685889638065035    ; 20 / ln 10
c_abs   dq 0x7fffffffffffffff

section .bss
alignb 8
clip    resb CL_SIZE

section .text

; xmm0 = linear level. prints it in dBFS
print_db:
    sub rsp, 40
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    ja .log
    say "-inf"
    jmp .out
.log:
    call math_log
    mulsd xmm0, [c_db]
    mov edx, 1
    call print_fixed
.out:
    say " dBFS"
    add rsp, 40
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    call lib_init
    cmp qword [argc], 2
    jae .go
    say "usage: wavinfo file.wav [more.wav ...]", 13, 10
    mov ecx, 1
    call ExitProcess
.go:
    mov r12d, 1
.file:
    lea rax, [argv]
    mov rcx, [rax+r12*8]
    mov [rsp+32], rcx
    call print_z
    say "  "
    mov rcx, [rsp+32]
    lea rdx, [clip]
    call wav_read
    test eax, eax
    jnz .read
    say "not a wav this can read", 13, 10
    jmp .nextfile
.read:
    mov ecx, [clip+CL_RATE]
    call print_dec
    say " Hz, "
    mov ecx, [clip+CL_CH]
    call print_dec
    say " ch, "
    cmp dword [clip+CL_FLOAT], 0
    je .int
    say "float"
    jmp .bits
.int:
    say "pcm"
.bits:
    mov ecx, [clip+CL_BITS]
    call print_dec
    say ", "
    mov rcx, [clip+CL_FRAMES]
    call print_dec
    say " frames = "
    cvtsi2sd xmm0, qword [clip+CL_FRAMES]
    cvtsi2sd xmm1, dword [clip+CL_RATE]
    divsd xmm0, xmm1
    mov edx, 3
    call print_fixed
    say " s", 13, 10

    ; peak and rms over every sample, in f64
    mov rsi, [clip+CL_DATA]
    mov rbx, [clip+CL_FRAMES]
    mov eax, [clip+CL_CH]
    imul rbx, rax
    mov rdi, rbx
    xorpd xmm2, xmm2            ; peak
    xorpd xmm3, xmm3            ; sum of squares
    movsd xmm4, [c_abs]
    test rbx, rbx
    jz .stats
.sample:
    cvtss2sd xmm0, [rsi]
    andpd xmm0, xmm4
    maxsd xmm2, xmm0
    mulsd xmm0, xmm0
    addsd xmm3, xmm0
    add rsi, 4
    dec rbx
    jnz .sample
    cvtsi2sd xmm0, rdi
    divsd xmm3, xmm0
    sqrtsd xmm3, xmm3
.stats:
    movsd [rsp+40], xmm3
    movsd [rsp+48], xmm2
    say "    peak "
    movsd xmm0, [rsp+48]
    call print_db
    say ", rms "
    movsd xmm0, [rsp+40]
    call print_db
    say 13, 10
    mov rcx, [clip+CL_DATA]
    call mem_free
.nextfile:
    inc r12d
    cmp r12, [argc]
    jb .file
    call con_restore
    xor ecx, ecx
    call ExitProcess
