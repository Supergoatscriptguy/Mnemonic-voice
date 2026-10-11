; stage 2 test: the resampler against a naive f64 version, streaming against one shot,
; then sines through it for gain, phase and aliasing
; uses: audio\resample
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "audio/audio.inc"

NX equ 100000
NY equ 160000

section .rdata
align 16
c_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff
c_one   dq 1.0
c_zero  dq 0.0
c_half  dq 0.5
c_2pi   dq 6.283185307179586
c_pi    dq 3.141592653589793
c_fc2   dq 0.96
c_beta  dq 8.95926
c_tiny  dq 1e-12
c_db10  dq 4.342944819032518    ; 10 / ln 10
c_tol   dq 2e-6
c_snr   dq 80.0
c_rej   dq -80.0

section .bss
alignb 32
rs      resb RS_SIZE
rs2     resb RS_SIZE
ra      resb RNG_SIZE
x       resd NX
y       resd NY
y2      resd NY
alignb 8
nx      resq 1                  ; inputs the naive version sees
ib      resq 1                  ; I0(beta)
cc      resq 1                  ; min(1, L/M)
hf      resq 1                  ; H as a double
sinf    resq 1                  ; sine test: frequency
sinin   resq 1                  ;   input rate
sinout  resq 1                  ;   output rate

section .text

; rcx = n. xmm0 = output n of x[0..nx) under rs's L, M, H, straight from the
; formula in f64: no tables, no phases, no streaming
naive:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 72
    ; locals: t +32, num +40, den +48, u +56, w +64
    mov rax, rcx
    mov ecx, [rs+RS_M]
    mul rcx
    mov ecx, [rs+RS_L]
    xor edx, edx
    div rcx
    mov r12, rax
    cvtsi2sd xmm0, rdx
    cvtsi2sd xmm1, rcx
    divsd xmm0, xmm1
    cvtsi2sd xmm2, rax
    addsd xmm2, xmm0
    movsd [rsp+32], xmm2
    xorpd xmm0, xmm0
    movsd [rsp+40], xmm0
    movsd [rsp+48], xmm0
    mov ebx, [rs+RS_H]
    mov rsi, r12
    sub rsi, rbx
    inc rsi                     ; first input, i - H + 1
    lea rdi, [r12+rbx]          ; last, i + H
.j:
    cvtsi2sd xmm0, rsi
    subsd xmm0, [rsp+32]
    movsd [rsp+56], xmm0
    divsd xmm0, [hf]
    mulsd xmm0, xmm0
    movsd xmm1, [c_one]
    subsd xmm1, xmm0
    maxsd xmm1, [c_zero]
    sqrtsd xmm0, xmm1
    mulsd xmm0, [c_beta]
    call bessel_i0
    divsd xmm0, [ib]
    movsd [rsp+64], xmm0
    movsd xmm0, [rsp+56]
    mulsd xmm0, [cc]
    mulsd xmm0, [c_fc2]
    mulsd xmm0, [c_pi]
    movapd xmm1, xmm0
    andpd xmm1, [c_abs]
    comisd xmm1, [c_tiny]
    jb .center
    movsd [rsp+56], xmm0
    call math_sin
    divsd xmm0, [rsp+56]
    jmp .h
.center:
    movsd xmm0, [c_one]
.h:
    mulsd xmm0, [rsp+64]
    movsd xmm1, [rsp+48]
    addsd xmm1, xmm0
    movsd [rsp+48], xmm1
    test rsi, rsi
    js .skip
    cmp rsi, [nx]
    jge .skip
    lea rax, [x]
    cvtss2sd xmm1, [rax+rsi*4]
    mulsd xmm0, xmm1
    addsd xmm0, [rsp+40]
    movsd [rsp+40], xmm0
.skip:
    inc rsi
    cmp rsi, rdi
    jle .j
    movsd xmm0, [rsp+40]
    divsd xmm0, [rsp+48]
    add rsp, 72
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = in rate, edx = out rate. inits rs and the naive version's constants
setup:
    sub rsp, 40
    mov r8d, edx
    mov edx, ecx
    lea rcx, [rs]
    call rs_init
    cvtsi2sd xmm0, dword [rs+RS_L]
    cvtsi2sd xmm1, dword [rs+RS_M]
    divsd xmm0, xmm1
    minsd xmm0, [c_one]
    movsd [cc], xmm0
    cvtsi2sd xmm0, dword [rs+RS_H]
    movsd [hf], xmm0
    movsd xmm0, [c_beta]
    call bessel_i0
    movsd [ib], xmm0
    add rsp, 40
    ret

; rs, all nx of x in one push and a flush, into y. rax = outputs
oneshot:
    push rbx
    sub rsp, 32
    lea rcx, [rs]
    lea rdx, [x]
    mov r8, [nx]
    lea r9, [y]
    call rs_push
    mov rbx, rax
    lea rcx, [rs]
    lea rdx, [y]
    lea rdx, [rdx+rbx*4]
    call rs_flush
    add rax, rbx
    add rsp, 32
    pop rbx
    ret

; prints "in -> out: L/M, taps", ecx = in, edx = out
title:
    push rbx
    push rsi
    sub rsp, 40
    mov ebx, ecx
    mov esi, edx
    say "  "
    mov ecx, ebx
    call print_dec
    say " -> "
    mov ecx, esi
    call print_dec
    say ": L/M = "
    mov ecx, [rs+RS_L]
    call print_dec
    say "/"
    mov ecx, [rs+RS_M]
    call print_dec
    say ", "
    mov ecx, [rs+RS_H]
    add ecx, ecx
    call print_dec
    say " taps", 13, 10
    add rsp, 40
    pop rsi
    pop rbx
    ret

; ecx = in, edx = out, r8 = inputs. noise through rs vs the naive version, and one
; shot vs random chunk sizes. prints and checks both
vsnaive:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 56
    mov [rsp+40], ecx
    mov [rsp+44], edx
    mov [nx], r8
    mov rbx, r8
    lea rsi, [x]
    xor edi, edi
.fill:
    lea rcx, [ra]
    call rng_float
    subsd xmm0, [c_half]
    cvtsd2ss xmm0, xmm0
    movss [rsi+rdi*4], xmm0
    inc rdi
    cmp rdi, rbx
    jb .fill
    mov ecx, [rsp+40]
    mov edx, [rsp+44]
    call setup
    mov ecx, [rsp+40]
    mov edx, [rsp+44]
    call title
    call oneshot
    mov r12, rax                ; outputs

    ; ceil(n L / M) of them
    mov rax, rbx
    mov ecx, [rs+RS_L]
    mul rcx
    mov ecx, [rs+RS_M]
    lea rax, [rax+rcx-1]
    xor edx, edx
    div rcx
    cmp rax, r12
    check e, "    ceil(n L / M) outputs"

    ; the same input pushed in random chunks of 1..1500, into y2
    lea rcx, [rs2]
    mov edx, [rsp+40]
    mov r8d, [rsp+44]
    call rs_init
    xor r13d, r13d              ; inputs pushed
    xor r14d, r14d              ; outputs so far
.chunk:
    mov rax, rbx
    sub rax, r13
    jz .flush
    mov [rsp+32], rax
    lea rcx, [ra]
    call rng_next
    xor edx, edx
    mov ecx, 1500
    div rcx
    lea r8, [rdx+1]
    cmp r8, [rsp+32]
    cmova r8, [rsp+32]
    mov rdi, r8
    lea rcx, [rs2]
    lea rdx, [x]
    lea rdx, [rdx+r13*4]
    lea r9, [y2]
    lea r9, [r9+r14*4]
    call rs_push
    add r14, rax
    add r13, rdi
    jmp .chunk
.flush:
    lea rcx, [rs2]
    lea rdx, [y2]
    lea rdx, [rdx+r14*4]
    call rs_flush
    add r14, rax
    xor eax, eax
    cmp r14, r12
    jne .diff
    lea rsi, [y]
    lea rdi, [y2]
    mov rcx, r12
    repe cmpsd
    sete al
.diff:
    cmp eax, 1
    check e, "    random chunks give the same bits as one push"
    lea rcx, [rs2]
    call rs_free

    ; every 3rd output against the naive f64 one
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    xor r13d, r13d
.cmp:
    mov rcx, r13
    call naive
    lea rax, [y]
    cvtss2sd xmm1, [rax+r13*4]
    subsd xmm0, xmm1
    andpd xmm0, [c_abs]
    maxsd xmm0, [rsp+32]
    movsd [rsp+32], xmm0
    add r13, 3
    cmp r13, r12
    jb .cmp
    movsd xmm0, [rsp+32]
    close_to 0.0, 2e-6, "    max |fast - naive f64|"
    lea rcx, [rs]
    call rs_free
    add rsp, 56
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; xmm0 = 0.5 sin(2 pi f n / rate), with the phase reduced exactly in integers.
; rcx = n, rdx = f, r8 = rate
sine:
    sub rsp, 40
    mov rax, rcx
    mul rdx
    div r8                      ; rdx = f n mod rate
    cvtsi2sd xmm0, rdx
    cvtsi2sd xmm1, r8
    divsd xmm0, xmm1
    mulsd xmm0, [c_2pi]
    call math_sin
    mulsd xmm0, [c_half]
    add rsp, 40
    ret

; ecx = in, edx = out, r8d = frequency, r9d = 1 if it's in the passband.
; one second of a 0.5 sine through rs. passband: snr against the ideal sine at the
; output rate (gain and phase both count). stopband: what's left, relative to the input
tone:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 64
    ; locals: signal power +32, error power +40, passband flag +48
    mov [rsp+48], r9d
    mov [sinin], rcx
    mov [sinout], rdx
    mov [sinf], r8
    mov [nx], rcx
    call setup
    xor ebx, ebx
.gen:
    mov rcx, rbx
    mov rdx, [sinf]
    mov r8, [sinin]
    call sine
    cvtsd2ss xmm0, xmm0
    lea rax, [x]
    movss [rax+rbx*4], xmm0
    inc rbx
    cmp rbx, [nx]
    jb .gen
    call oneshot
    mov r12, rax
    ; skip the edges, where the taps hang off the signal
    mov eax, [rs+RS_H]
    mov ecx, [rs+RS_L]
    mul rcx
    mov ecx, [rs+RS_M]
    div rcx
    lea rbx, [rax+2]
    mov r13, r12
    sub r13, rbx
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    movsd [rsp+40], xmm0
.acc:
    mov rcx, rbx
    mov rdx, [sinf]
    mov r8, [sinout]
    call sine
    lea rax, [y]
    cvtss2sd xmm1, [rax+rbx*4]
    cmp dword [rsp+48], 0
    jne .vsideal
    movapd xmm2, xmm1           ; stopband: everything that came out is error,
    mulsd xmm2, xmm2            ; the signal is what went in
    jmp .sums
.vsideal:
    subsd xmm1, xmm0
    movapd xmm2, xmm1
    mulsd xmm2, xmm2
.sums:
    mulsd xmm0, xmm0
    addsd xmm0, [rsp+32]
    movsd [rsp+32], xmm0
    addsd xmm2, [rsp+40]
    movsd [rsp+40], xmm2
    inc rbx
    cmp rbx, r13
    jb .acc

    say "  "
    mov rcx, [sinin]
    call print_dec
    say " -> "
    mov rcx, [sinout]
    call print_dec
    say ", "
    mov rcx, [sinf]
    call print_dec
    ; error over signal, in dB
    movsd xmm0, [rsp+40]
    divsd xmm0, [rsp+32]
    call math_log
    mulsd xmm0, [c_db10]
    cmp dword [rsp+48], 0
    je .stop
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
    movsd [rsp+56], xmm0
    say " Hz (passband)"
    movsd xmm0, [rsp+56]
    comisd xmm0, [c_snr]
    seta cl
    movzx ecx, cl
    movapd xmm2, xmm0
    lea rdx, [s_snr]
    call t_okf
    jmp .done
.stop:
    movsd [rsp+56], xmm0
    say " Hz (stopband)"
    movsd xmm0, [rsp+56]
    movsd xmm1, [c_rej]
    comisd xmm1, xmm0
    seta cl
    movzx ecx, cl
    movapd xmm2, xmm0
    lea rdx, [s_rej]
    call t_okf
.done:
    lea rcx, [rs]
    call rs_free
    add rsp, 64
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
s_snr   db "  snr > 80 dB", 0
s_rej   db "  left over < -80 dB", 0
c_tenm  dq 10.0
c_rt    dq 20.833333333333332   ; seconds in 1M samples at 48k
section .text

%macro tonecheck 4
    mov ecx, %1
    mov edx, %2
    mov r8d, %3
    mov r9d, %4
    call tone
%endmacro

global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init
    lea rcx, [ra]
    mov edx, 11
    call rng_seed

    say "bessel I0", 13, 10
    xorpd xmm0, xmm0
    call bessel_i0
    close_to 1.0, 1e-15, "  I0(0)"
    movsd xmm0, [c_one]
    call bessel_i0
    close_to 1.2660658777520082, 1e-15, "  I0(1)"
    movsd xmm0, [c_tenm]
    call bessel_i0
    close_to 2815.716628466254, 1e-9, "  I0(10)"

    say "fast vs naive, streaming vs one shot (noise)", 13, 10
    mov ecx, 44100
    mov edx, 16000
    mov r8d, 20000
    call vsnaive
    mov ecx, 48000
    mov edx, 16000
    mov r8d, 12000
    call vsnaive
    mov ecx, 16000
    mov edx, 48000
    mov r8d, 4000
    call vsnaive
    mov ecx, 22050
    mov edx, 24000
    mov r8d, 8000
    call vsnaive

    say "sines", 13, 10
    tonecheck 48000, 16000, 1000, 1
    tonecheck 48000, 16000, 7000, 1
    tonecheck 48000, 16000, 8600, 0
    tonecheck 48000, 16000, 12000, 0
    tonecheck 44100, 16000, 1000, 1
    tonecheck 44100, 16000, 9000, 0
    tonecheck 16000, 48000, 1000, 1
    tonecheck 16000, 48000, 7000, 1
    tonecheck 22050, 24000, 1000, 1
    tonecheck 22050, 24000, 10000, 1
    tonecheck 24000, 48000, 1000, 1
    tonecheck 48000, 24000, 13000, 0

    say "speed", 13, 10
    mov ecx, 48000
    mov edx, 16000
    call setup
    call time_now
    mov [rsp+32], rax
    mov ebx, 10
.rep:
    lea rcx, [rs]
    lea rdx, [x]
    mov r8d, NX
    lea r9, [y]
    call rs_push
    dec ebx
    jnz .rep
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    say "  48000 -> 16000, 1M samples (20.8 s of audio): "
    movsd xmm0, [rsp+40]
    mov edx, 4
    call print_fixed
    say " s, "
    movsd xmm0, [c_rt]
    divsd xmm0, [rsp+40]
    xor edx, edx
    call print_fixed
    say "x real time on one thread", 13, 10
    jmp t_done
