; sample rate conversion for any rational ratio L/M (out/in), polyphase windowed sinc.
; output n sits at input time n M / L. its taps are a sinc cut at 0.92 of the lower
; nyquist under a kaiser window (beta 8.96, ~90 dB), so the stopband starts right at
; the lower nyquist and nothing above it folds back. each phase sums to exactly 1.
; mono, streaming: push chunks of any size, flush at the end. avx2 + fma
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

section .rdata
align 16
c_abs    dq 0x7fffffffffffffff, 0x7fffffffffffffff
c_one    dq 1.0
c_zero   dq 0.0
c_quart  dq 0.25
c_halfw  dq 71.4                ; half length at c = 1 for a 0.04 c transition at 90 dB
c_fc2    dq 0.96                ; sinc zero spacing: cutoff 0.48 c cycles per input sample
c_beta   dq 8.95926             ; 0.1102 (90 - 8.7)
c_pi     dq 3.141592653589793
c_tiny   dq 1e-12
c_eps    dq 1e-17

section .text

; xmm0 = x. xmm0 = I0(x), straight from the power series. touches xmm0-4
global bessel_i0
bessel_i0:
    mulsd xmm0, xmm0
    mulsd xmm0, [c_quart]       ; x^2 / 4
    movsd xmm1, [c_one]         ; sum
    movsd xmm2, xmm1            ; term
    movsd xmm3, xmm1            ; k
.term:
    mulsd xmm2, xmm0            ; term *= (x^2/4) / k^2
    movapd xmm4, xmm3
    mulsd xmm4, xmm3
    divsd xmm2, xmm4
    addsd xmm1, xmm2
    addsd xmm3, [c_one]
    movapd xmm4, xmm1
    mulsd xmm4, [c_eps]
    comisd xmm2, xmm4
    ja .term
    movapd xmm0, xmm1
    ret

; rcx = rs, edx = input rate, r8d = output rate
global rs_init
rs_init:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96
    ; locals: c +32, I0(beta) +40, frac +48, sum +56, u +64, f64 row +72, H +80, window +88
    mov rbx, rcx
    mov r12d, edx
    mov r13d, r8d
    mov eax, edx
    mov ecx, r8d
.gcd:
    test ecx, ecx
    jz .gotgcd
    xor edx, edx
    div ecx
    mov eax, ecx
    mov ecx, edx
    jmp .gcd
.gotgcd:
    mov ecx, eax
    mov eax, r13d
    xor edx, edx
    div ecx
    mov [rbx+RS_L], eax
    mov r14d, eax
    mov eax, r12d
    xor edx, edx
    div ecx
    mov [rbx+RS_M], eax
    mov r15d, eax

    ; c = min(1, L/M), the lower nyquist as a fraction of the input's
    cvtsi2sd xmm0, r14
    cvtsi2sd xmm1, r15
    divsd xmm0, xmm1
    minsd xmm0, [c_one]
    movsd [rsp+32], xmm0
    movsd xmm1, [c_halfw]
    divsd xmm1, xmm0
    roundsd xmm1, xmm1, 2       ; ceil
    cvtsd2si eax, xmm1
    mov [rbx+RS_H], eax
    mov esi, eax
    lea eax, [rsi*2+7]
    and eax, -8
    mov [rbx+RS_TAPS], eax
    mov edi, eax

    mov ecx, r14d
    imul ecx, edi
    shl rcx, 2
    call mem_alloc
    mov [rbx+RS_TAB], rax
    lea ecx, [rsi*2]
    shl rcx, 3
    call mem_alloc
    mov [rsp+72], rax
    movsd xmm0, [c_beta]
    call bessel_i0
    movsd [rsp+40], xmm0
    cvtsi2sd xmm0, rsi
    movsd [rsp+80], xmm0

    xor r12d, r12d              ; phase
.phase:
    cvtsi2sd xmm0, r12
    cvtsi2sd xmm1, r14
    divsd xmm0, xmm1
    movsd [rsp+48], xmm0
    xorpd xmm0, xmm0
    movsd [rsp+56], xmm0
    xor r13d, r13d              ; tap
.tap:
    ; u = tap - H + 1 - phase/L, in input samples from the output's time
    mov eax, r13d
    sub eax, esi
    inc eax
    cvtsi2sd xmm0, eax
    subsd xmm0, [rsp+48]
    movsd [rsp+64], xmm0
    divsd xmm0, [rsp+80]
    mulsd xmm0, xmm0
    movsd xmm1, [c_one]
    subsd xmm1, xmm0
    maxsd xmm1, [c_zero]
    sqrtsd xmm0, xmm1
    mulsd xmm0, [c_beta]
    call bessel_i0
    divsd xmm0, [rsp+40]
    movsd [rsp+88], xmm0
    ; sinc: sin(pi x) / (pi x), x = 0.96 c u
    movsd xmm0, [rsp+64]
    mulsd xmm0, [rsp+32]
    mulsd xmm0, [c_fc2]
    mulsd xmm0, [c_pi]
    movapd xmm1, xmm0
    andpd xmm1, [c_abs]
    comisd xmm1, [c_tiny]
    jb .center
    movsd [rsp+64], xmm0
    call math_sin
    divsd xmm0, [rsp+64]
    jmp .windowed
.center:
    movsd xmm0, [c_one]
.windowed:
    mulsd xmm0, [rsp+88]
    mov rax, [rsp+72]
    movsd [rax+r13*8], xmm0
    addsd xmm0, [rsp+56]
    movsd [rsp+56], xmm0
    inc r13d
    lea eax, [rsi*2]
    cmp r13d, eax
    jb .tap
    ; normalize into this phase's f32 row (the padding past 2H stays 0)
    mov rdx, r12
    imul rdx, rdi
    shl rdx, 2
    add rdx, [rbx+RS_TAB]
    mov rcx, [rsp+72]
    movsd xmm1, [rsp+56]
    xor r13d, r13d
.norm:
    movsd xmm0, [rcx+r13*8]
    divsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    movss [rdx+r13*4], xmm0
    inc r13d
    lea eax, [rsi*2]
    cmp r13d, eax
    jb .norm
    inc r12d
    cmp r12d, r14d
    jb .phase
    mov rcx, [rsp+72]
    call mem_free

    ; H - 1 zeros of history sit in front of t = 0
    lea ecx, [rsi*2+16384]
    add ecx, edi
    mov [rbx+RS_CAP], rcx
    shl rcx, 2
    call mem_alloc
    mov [rbx+RS_BUF], rax
    lea eax, [rsi-1]
    mov [rbx+RS_HAVE], rax
    xor eax, eax
    mov [rbx+RS_REL], rax
    mov [rbx+RS_PH], eax
    mov [rbx+RS_NIN], rax
    mov [rbx+RS_NOUT], rax
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = rs, rdx = n. grows the buffer so n more samples (plus a row's padding) fit
room:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, rcx
    mov rax, [rbx+RS_HAVE]
    add rax, rdx
    mov ecx, [rbx+RS_TAPS]
    add rax, rcx
    cmp rax, [rbx+RS_CAP]
    jbe .fits
    add rax, rax
    mov [rbx+RS_CAP], rax
    lea rcx, [rax*4]
    call mem_alloc
    mov rdi, rax
    mov rsi, [rbx+RS_BUF]
    mov rcx, [rbx+RS_HAVE]
    rep movsd
    mov rcx, [rbx+RS_BUF]
    mov [rbx+RS_BUF], rax
    call mem_free
.fits:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = rs, rdx = out, r8 = most outputs to make. makes every output whose taps
; are all in, then drops the samples no later output needs. rax = outputs
produce:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    xor r14d, r14d
    mov rsi, [rbx+RS_REL]
    mov edi, [rbx+RS_PH]
    mov r15, [rbx+RS_BUF]
    mov r8d, [rbx+RS_H]
    add r8d, r8d
    mov r9d, [rbx+RS_TAPS]
    mov r10, [rbx+RS_TAB]
    mov r11, [rbx+RS_HAVE]
.out:
    cmp r14, r13
    jae .done
    lea rax, [rsi+r8]
    cmp rax, r11
    ja .done
    lea rcx, [r15+rsi*4]
    mov rdx, rdi
    imul rdx, r9
    lea rdx, [r10+rdx*4]
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    mov eax, r9d
.dot16:
    cmp eax, 16
    jb .dot8
    vmovups ymm2, [rcx]
    vfmadd231ps ymm0, ymm2, [rdx]
    vmovups ymm3, [rcx+32]
    vfmadd231ps ymm1, ymm3, [rdx+32]
    add rcx, 64
    add rdx, 64
    sub eax, 16
    jmp .dot16
.dot8:
    test eax, eax
    jz .hsum
    vmovups ymm2, [rcx]
    vfmadd231ps ymm0, ymm2, [rdx]
.hsum:
    vaddps ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vmovshdup xmm1, xmm0
    vaddss xmm0, xmm0, xmm1
    vmovss [r12+r14*4], xmm0
    inc r14
    ; next output: the phase moves M/L of an input sample, whole samples carry into rel
    add edi, [rbx+RS_M]
.carry:
    cmp edi, [rbx+RS_L]
    jb .out
    sub edi, [rbx+RS_L]
    inc rsi
    jmp .carry
.done:
    vzeroupper
    mov [rbx+RS_PH], edi
    add [rbx+RS_NOUT], r14
    mov rcx, r11
    sub rcx, rsi
    mov [rbx+RS_HAVE], rcx
    mov qword [rbx+RS_REL], 0
    lea rsi, [r15+rsi*4]
    mov rdi, r15
    rep movsd
    mov rax, r14
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = rs, rdx = samples, r8 = how many, r9 = out (room for rs_maxout(n)).
; rax = outputs written
global rs_push
rs_push:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    mov rbx, rcx
    mov rsi, rdx
    mov r12, r8
    mov r13, r9
    add [rbx+RS_NIN], r8
    mov rdx, r8
    call room
    mov rdi, [rbx+RS_HAVE]
    shl rdi, 2
    add rdi, [rbx+RS_BUF]
    mov rcx, r12
    rep movsd
    add [rbx+RS_HAVE], r12
    mov rcx, rbx
    mov rdx, r13
    mov r8, -1
    call produce
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = rs, rdx = out. finishes the stream: exactly ceil(inputs L / M) outputs in
; all, the last ones made against trailing zeros. rax = outputs written
global rs_flush
rs_flush:
    push rbx
    push rdi
    push r12
    sub rsp, 48
    mov rbx, rcx
    mov r12, rdx
    mov rax, [rbx+RS_NIN]
    mov ecx, [rbx+RS_L]
    mul rcx
    mov ecx, [rbx+RS_M]
    lea rax, [rax+rcx-1]
    xor edx, edx
    div rcx
    sub rax, [rbx+RS_NOUT]
    jbe .none
    mov [rsp+32], rax           ; outputs still owed
    mov rcx, rbx
    mov edx, [rbx+RS_H]
    inc edx
    call room
    mov rdi, [rbx+RS_HAVE]
    shl rdi, 2
    add rdi, [rbx+RS_BUF]
    mov ecx, [rbx+RS_H]
    inc ecx
    add [rbx+RS_HAVE], rcx
    xor eax, eax
    rep stosd
    mov rcx, rbx
    mov rdx, r12
    mov r8, [rsp+32]
    call produce
    jmp .out
.none:
    xor eax, eax
.out:
    add rsp, 48
    pop r12
    pop rdi
    pop rbx
    ret

; rcx = rs, rdx = n inputs. rax = the most outputs pushing them can make
global rs_maxout
rs_maxout:
    mov rax, rdx
    mov r8d, [rcx+RS_L]
    mul r8
    mov r8d, [rcx+RS_M]
    div r8
    add rax, 2
    ret

; rcx = rs
global rs_free
rs_free:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    mov rcx, [rbx+RS_TAB]
    call mem_free
    mov rcx, [rbx+RS_BUF]
    call mem_free
    add rsp, 32
    pop rbx
    ret

; rcx = clip, edx = new rate, r8 = clip to fill (not the same one). each channel
; goes through its own resampler. the new CL_DATA is mem_alloc'd. same rate is a copy
global clip_resample
clip_resample:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 144
    ; locals: resampler +32 (RS_SIZE), mono in +112, mono out +120
    mov rbx, rcx
    mov r12d, edx
    mov r13, r8
    mov [r13+CL_RATE], edx
    mov r14d, [rbx+CL_CH]
    mov [r13+CL_CH], r14d
    mov dword [r13+CL_BITS], 32
    mov dword [r13+CL_FLOAT], 1
    mov r15, [rbx+CL_FRAMES]
    cmp edx, [rbx+CL_RATE]
    jne .convert
    mov [r13+CL_FRAMES], r15
    mov rcx, r15
    imul rcx, r14
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov [r13+CL_DATA], rax
    mov rdi, rax
    mov rsi, [rbx+CL_DATA]
    mov rcx, r15
    imul rcx, r14
    rep movsd
    jmp .out

.convert:
    lea rcx, [r15*4+64]
    call mem_alloc
    mov [rsp+112], rax
    xor esi, esi                ; channel
.chan:
    lea rcx, [rsp+32]
    mov edx, [rbx+CL_RATE]
    mov r8d, r12d
    call rs_init
    test esi, esi
    jnz .sized
    ; the first channel's resampler knows L and M: ceil(frames L / M) out
    mov rax, r15
    mov ecx, [rsp+32+RS_L]
    mul rcx
    mov ecx, [rsp+32+RS_M]
    lea rax, [rax+rcx-1]
    xor edx, edx
    div rcx
    mov [r13+CL_FRAMES], rax
    mov rdi, rax
    imul rax, r14
    lea rcx, [rax*4+64]
    call mem_alloc
    mov [r13+CL_DATA], rax
    lea rcx, [rdi*4+64]
    call mem_alloc
    mov [rsp+120], rax
.sized:
    mov rdx, [rbx+CL_DATA]
    lea rdx, [rdx+rsi*4]
    mov rdi, [rsp+112]
    mov rcx, r15
    test rcx, rcx
    jz .split
.take:
    mov eax, [rdx]
    mov [rdi], eax
    lea rdx, [rdx+r14*4]
    add rdi, 4
    dec rcx
    jnz .take
.split:
    lea rcx, [rsp+32]
    mov rdx, [rsp+112]
    mov r8, r15
    mov r9, [rsp+120]
    call rs_push
    mov rdi, rax
    lea rcx, [rsp+32]
    mov rdx, [rsp+120]
    lea rdx, [rdx+rdi*4]
    call rs_flush
    mov rcx, [r13+CL_FRAMES]
    mov rdx, [rsp+120]
    mov rdi, [r13+CL_DATA]
    lea rdi, [rdi+rsi*4]
    test rcx, rcx
    jz .merged
.give:
    mov eax, [rdx]
    mov [rdi], eax
    add rdx, 4
    lea rdi, [rdi+r14*4]
    dec rcx
    jnz .give
.merged:
    lea rcx, [rsp+32]
    call rs_free
    inc esi
    cmp esi, r14d
    jb .chan
    mov rcx, [rsp+112]
    call mem_free
    mov rcx, [rsp+120]
    call mem_free
.out:
    add rsp, 144
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
