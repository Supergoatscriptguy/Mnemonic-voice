; stage 3 test: the fft and log-mel against a naive f64 version written straight from
; the formulas (a plain dft, the slaney triangles), plus values known by hand
; uses: audio\mel
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "audio/audio.inc"

NS      equ 4000                ; 25 frames of test signal

section .rdata
align 8
c_2pi   dq 6.283185307179586
c_half  dq 0.5
c_one   dq 1.0
c_two   dq 2.0
c_zero  dq 0.0
c_400   dq 400.0
c_512   dq 512.0
c_binhz dq 31.25
c_eps   dq 1e-6
c_1k    dq 1000.0
c_500   dq 500.0
c_3k    dq 3000.0
c_8k    dq 8000.0
align 16
c_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff
c_f440  dq 0.17278759594743864  ; 2 pi 440 / 16000
c_f2k   dq 0.7853981633974483   ; 2 pi 2000 / 16000
c_f1k   dq 0.39269908169872414  ; 2 pi 1000 / 16000 = bin 32
c_m     dq 1e-6

section .bss
alignb 32
mel     resb MEL_SIZE
sig     resd NS
pw      resd 260
ref     resq 260                ; naive power spectrum, f64
wts     resq 80 * 257           ; naive filterbank, f64
edges   resq 82
feats   resd 26 * 80
ra      resb RNG_SIZE

section .text

; rcx = f32 frame (400 samples, zeros past r8d of them), edx = ignored, r8d = valid.
; ref[0..256] = |X_k|^2 by a plain dft of the hann windowed, zero padded frame, in f64
naive_power:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 72
    ; locals: re sum +32, im sum +40, sample +48, angle +56
    mov rsi, rcx
    mov r12d, r8d
    xor ebx, ebx                ; k
.k:
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    movsd [rsp+40], xmm0
    xor edi, edi                ; n
.n:
    cmp edi, r12d
    jae .sum
    cmp edi, 400
    jae .sum
    ; x[n] w[n], w = 0.5 - 0.5 cos(2 pi n / 400)
    cvtsi2sd xmm0, edi
    mulsd xmm0, [c_2pi]
    divsd xmm0, [c_400]
    call math_cos
    mulsd xmm0, [c_half]
    movsd xmm1, [c_half]
    subsd xmm1, xmm0
    cvtss2sd xmm0, [rsi+rdi*4]
    mulsd xmm0, xmm1
    movsd [rsp+48], xmm0
    ; angle 2 pi k n / 512
    mov eax, ebx
    imul eax, edi
    cvtsi2sd xmm0, eax
    mulsd xmm0, [c_2pi]
    divsd xmm0, [c_512]
    movsd [rsp+56], xmm0
    call math_cos
    mulsd xmm0, [rsp+48]
    addsd xmm0, [rsp+32]
    movsd [rsp+32], xmm0
    movsd xmm0, [rsp+56]
    call math_sin
    mulsd xmm0, [rsp+48]
    movsd xmm1, [rsp+40]
    subsd xmm1, xmm0
    movsd [rsp+40], xmm1
    inc edi
    jmp .n
.sum:
    movsd xmm0, [rsp+32]
    mulsd xmm0, xmm0
    movsd xmm1, [rsp+40]
    mulsd xmm1, xmm1
    addsd xmm0, xmm1
    lea rax, [ref]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, 257
    jb .k
    add rsp, 72
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the slaney filterbank from its definition into wts[80][257], f64
naive_bank:
    push rbx
    push rsi
    push rdi
    sub rsp, 64
    movsd xmm0, [c_8k]
    call hz2mel
    movsd [rsp+32], xmm0
    xor ebx, ebx
.e:
    cvtsi2sd xmm0, ebx
    mulsd xmm0, [rsp+32]
    mov eax, 81
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    call mel2hz
    lea rax, [edges]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, 82
    jb .e
    xor ebx, ebx                ; filter
.f:
    lea rax, [edges]
    movsd xmm2, [rax+rbx*8]
    movsd xmm3, [rax+rbx*8+8]
    movsd xmm4, [rax+rbx*8+16]
    xor esi, esi                ; bin
.b:
    cvtsi2sd xmm0, esi
    mulsd xmm0, [c_binhz]       ; f
    ; up = (f - lo) / (c - lo), down = (hi - f) / (hi - c)
    movapd xmm1, xmm0
    subsd xmm1, xmm2
    movapd xmm5, xmm3
    subsd xmm5, xmm2
    divsd xmm1, xmm5
    movapd xmm5, xmm4
    subsd xmm5, xmm0
    movapd xmm0, xmm4
    subsd xmm0, xmm3
    divsd xmm5, xmm0
    minsd xmm1, xmm5
    maxsd xmm1, [c_zero]
    ; times 2 / (hi - lo)
    movapd xmm0, xmm4
    subsd xmm0, xmm2
    movsd xmm5, [c_two]
    divsd xmm5, xmm0
    mulsd xmm1, xmm5
    imul eax, ebx, 257
    add eax, esi
    lea rdx, [wts]
    movsd [rdx+rax*8], xmm1
    inc esi
    cmp esi, 257
    jb .b
    inc ebx
    cmp ebx, 80
    jb .f
    add rsp, 64
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = f32 frame, r8d = valid samples. runs mel_power into pw and the naive dft
; into ref. xmm0 = the largest |fast - naive| over the largest naive value
power_vs:
    push rbx
    push rsi
    sub rsp, 40
    mov rsi, rcx
    mov ebx, r8d
    lea rcx, [mel]
    mov rdx, rsi
    lea r9, [pw]
    call mel_power
    mov rcx, rsi
    mov r8d, ebx
    call naive_power
    xorpd xmm2, xmm2            ; largest diff
    xorpd xmm3, xmm3            ; largest value
    xor ecx, ecx
    lea rdx, [pw]
    lea r8, [ref]
.k:
    cvtss2sd xmm0, [rdx+rcx*4]
    movsd xmm1, [r8+rcx*8]
    maxsd xmm3, xmm1
    subsd xmm0, xmm1
    andpd xmm0, [c_abs]
    maxsd xmm2, xmm0
    inc ecx
    cmp ecx, 257
    jb .k
    divsd xmm2, xmm3
    movapd xmm0, xmm2
    add rsp, 40
    pop rsi
    pop rbx
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    call lib_init
    lea rcx, [mel]
    call mel_init

    say "slaney mel scale", 13, 10
    movsd xmm0, [c_1k]
    call hz2mel
    close_to 15.0, 1e-12, "  mel(1000 Hz) = 15"
    movsd xmm0, [c_500]
    call hz2mel
    close_to 7.5, 1e-12, "  mel(500 Hz) = 7.5 (linear below 1 kHz)"
    movsd xmm0, [c_3k]
    call hz2mel
    call mel2hz
    close_to 3000.0, 1e-9, "  hz(mel(3000)) = 3000"

    say "filterbank vs the definition", 13, 10
    call naive_bank
    ; expand the sparse tables and compare
    xorpd xmm2, xmm2
    xor ebx, ebx                ; filter
    xor r12d, r12d              ; filters with no bins
.fb:
    lea rax, [mel]
    movzx esi, word [rax+MEL_FB0+rbx*2]
    movzx edi, word [rax+MEL_FBN+rbx*2]
    mov r8d, [rax+MEL_FBW+rbx*4]
    test edi, edi
    jnz .bins
    inc r12d
.bins:
    xor ecx, ecx                ; every bin
.cb:
    xorpd xmm0, xmm0
    mov edx, ecx
    sub edx, esi
    js .cmp
    cmp edx, edi
    jae .cmp
    add edx, r8d
    lea rax, [mel]
    cvtss2sd xmm0, [rax+MEL_W+rdx*4]
.cmp:
    imul eax, ebx, 257
    add eax, ecx
    lea rdx, [wts]
    subsd xmm0, [rdx+rax*8]
    andpd xmm0, [c_abs]
    maxsd xmm2, xmm0
    inc ecx
    cmp ecx, 257
    jb .cb
    inc ebx
    cmp ebx, 80
    jb .fb
    movapd xmm0, xmm2
    close_to 0.0, 1e-9, "  every weight, max |table - definition|"
    cmp r12d, 0
    check e, "  every filter covers at least one bin"

    say "fft (power spectrum) vs a plain dft", 13, 10
    ; an impulse at the window's peak: |X_k| = 1 everywhere
    lea rdi, [sig]
    xor eax, eax
    mov ecx, NS
    rep stosd
    lea rax, [sig]
    mov dword [rax+200*4], 0x3f800000
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, 400
    lea r9, [pw]
    call mel_power
    xorpd xmm2, xmm2
    xor ecx, ecx
    lea rdx, [pw]
.imp:
    cvtss2sd xmm0, [rdx+rcx*4]
    subsd xmm0, [c_one]
    andpd xmm0, [c_abs]
    maxsd xmm2, xmm0
    inc ecx
    cmp ecx, 257
    jb .imp
    movapd xmm0, xmm2
    close_to 0.0, 1e-6, "  impulse at sample 200: power 1 in every bin"
    ; a cosine at exactly bin 32 (1 kHz) peaks at bin 32
    xor ebx, ebx
.cos:
    cvtsi2sd xmm0, ebx
    mulsd xmm0, [c_f1k]
    call math_cos
    cvtsd2ss xmm0, xmm0
    lea rax, [sig]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, 400
    jb .cos
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, 400
    lea r9, [pw]
    call mel_power
    xor ebx, ebx
    xor ecx, ecx
    lea rdx, [pw]
    movss xmm1, [rdx]
.peak:
    movss xmm0, [rdx+rcx*4]
    comiss xmm0, xmm1
    jbe .np
    movss xmm1, xmm0
    mov ebx, ecx
.np:
    inc ecx
    cmp ecx, 257
    jb .peak
    cmp ebx, 32
    check e, "  1 kHz cosine peaks in bin 32"
    ; random frames, full and cut short
    lea rcx, [ra]
    mov edx, 3
    call rng_seed
    xor ebx, ebx
.rnd:
    lea rcx, [ra]
    call rng_float
    subsd xmm0, [c_half]
    addsd xmm0, xmm0
    cvtsd2ss xmm0, xmm0
    lea rax, [sig]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NS
    jb .rnd
    lea rcx, [sig]
    mov r8d, 400
    call power_vs
    close_to 0.0, 2e-6, "  noise frame, max |fast - dft| / max"
    ; parseval on the same frame: 512 sum (x w)^2 = |X0|^2 + |X256|^2 + 2 sum |Xk|^2
    xorpd xmm2, xmm2
    xor ecx, ecx
    lea rdx, [mel]
    lea r8, [sig]
.ps:
    cvtss2sd xmm0, [r8+rcx*4]
    cvtss2sd xmm1, [rdx+MEL_WIN+rcx*4]
    mulsd xmm0, xmm1
    mulsd xmm0, xmm0
    addsd xmm2, xmm0
    inc ecx
    cmp ecx, 400
    jb .ps
    mulsd xmm2, [c_512]
    lea rdx, [pw]
    cvtss2sd xmm3, [rdx]
    cvtss2sd xmm0, [rdx+256*4]
    addsd xmm3, xmm0
    xorpd xmm4, xmm4
    mov ecx, 1
.pk:
    cvtss2sd xmm0, [rdx+rcx*4]
    addsd xmm4, xmm0
    inc ecx
    cmp ecx, 256
    jb .pk
    addsd xmm4, xmm4
    addsd xmm3, xmm4
    divsd xmm3, xmm2
    movapd xmm0, xmm3
    close_to 1.0, 1e-5, "  parseval (spectrum energy / time energy)"
    lea rcx, [sig+1000*4]
    mov r8d, 230
    call power_vs
    close_to 0.0, 2e-6, "  a frame with only 230 samples left"

    say "log-mel, mel_run vs dft + definition + ln", 13, 10
    ; 0.25 s: two tones over quiet noise
    xor ebx, ebx
.mix:
    cvtsi2sd xmm0, ebx
    mulsd xmm0, [c_f440]
    call math_sin
    movsd [rsp+32], xmm0
    cvtsi2sd xmm0, ebx
    mulsd xmm0, [c_f2k]
    call math_sin
    mulsd xmm0, [c_half]
    addsd xmm0, [rsp+32]
    mulsd xmm0, [c_half]
    lea rax, [sig]
    cvtss2sd xmm1, [rax+rbx*4]
    mulsd xmm1, [c_m]           ; the noise from before, way down
    mulsd xmm1, [c_400]
    addsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NS
    jb .mix
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, NS
    lea r9, [feats]
    call mel_run
    cmp rax, 25
    check e, "  4000 samples = 25 frames"
    xorpd xmm0, xmm0
    movsd [rsp+40], xmm0        ; largest diff
    xor r12d, r12d              ; frame
.mf:
    imul eax, r12d, MEL_HOP
    lea rcx, [sig]
    lea rcx, [rcx+rax*4]
    mov r8d, NS
    sub r8d, eax
    cmp r8d, 400
    jbe .clip
    mov r8d, 400
.clip:
    call naive_power
    xor ebx, ebx                ; filter
.mm:
    xorpd xmm2, xmm2
    xor ecx, ecx
    imul eax, ebx, 257
    lea rdx, [wts]
    lea rdx, [rdx+rax*8]
    lea r8, [ref]
.dot:
    movsd xmm0, [rdx+rcx*8]
    mulsd xmm0, [r8+rcx*8]
    addsd xmm2, xmm0
    inc ecx
    cmp ecx, 257
    jb .dot
    addsd xmm2, [c_eps]
    movapd xmm0, xmm2
    call math_log
    imul eax, r12d, MEL_N
    add eax, ebx
    lea rdx, [feats]
    cvtss2sd xmm1, [rdx+rax*4]
    subsd xmm0, xmm1
    andpd xmm0, [c_abs]
    maxsd xmm0, [rsp+40]
    movsd [rsp+40], xmm0
    inc ebx
    cmp ebx, 80
    jb .mm
    inc r12d
    cmp r12d, 25
    jb .mf
    movsd xmm0, [rsp+40]
    close_to 0.0, 1e-3, "  25 frames x 80 filters, max |fast - naive| in ln units (f32 vs f64)"
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, 4001
    lea r9, [feats]
    call mel_run
    cmp rax, 26
    check e, "  4001 samples = 26 frames (the last one mostly zeros)"
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, 100
    lea r9, [feats]
    call mel_run
    cmp rax, 1
    check e, "  100 samples = 1 frame"

    say "speed", 13, 10
    call time_now
    mov [rsp+32], rax
    mov ebx, 400
.sp:
    lea rcx, [mel]
    lea rdx, [sig]
    mov r8d, NS
    lea r9, [feats]
    call mel_run
    dec ebx
    jnz .sp
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    say "  10000 frames (100 s of audio) in "
    movsd xmm0, [rsp+40]
    mov edx, 3
    call print_fixed
    say " s on one thread", 13, 10
    jmp t_done
