; stage 4 test: the general fft (audio\fft). 1024 points against a plain f64 dft,
; a 2^20 point round trip, and fft_conv against direct convolution
; uses: audio\fft
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"

extern fft_init, fft, fft_conv

N1      equ 1024
LBIG    equ 20
NX      equ 3000
NH      equ 500
LCONV   equ 12                  ; 4096 >= 3000 + 500 - 1

section .rdata
align 8
c_2pi   dq 6.283185307179586
c_ms    dq 1000.0
align 16
c_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff

section .bss
alignb 16
rng     resb RNG_SIZE
a       resd 2 * N1
a0      resd 2 * N1
big     resq 1
big0    resq 1
x       resd NX
h       resd NH
y       resd NX
scr     resd 2 * 4096

section .text

; xmm0 = a normal number as f32
rnorm:
    sub rsp, 40
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
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
    mov ecx, LBIG
    call fft_init
    lea rcx, [rng]
    mov edx, 3
    call rng_seed

    ; 1024 points against the dft
    xor ebx, ebx
.r1:
    call rnorm
    lea rax, [a]
    movss [rax+rbx*4], xmm0
    lea rax, [a0]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, 2 * N1
    jb .r1
    lea rcx, [a]
    mov edx, 10
    xor r8d, r8d
    call fft
    xorpd xmm6, xmm6            ; max diff
    xorpd xmm7, xmm7            ; max size
    xor r12d, r12d              ; k
.k:
    xorpd xmm4, xmm4            ; sum re
    xorpd xmm5, xmm5            ; sum im
    xor ebx, ebx                ; n
.n:
    ; e^(-2 pi i k n / N)
    mov eax, r12d
    imul eax, ebx
    and eax, N1 - 1
    cvtsi2sd xmm0, eax
    mulsd xmm0, [c_2pi]
    mov eax, N1
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+32], xmm0
    movsd [rsp+40], xmm4
    movsd [rsp+48], xmm5
    call math_cos
    movsd [rsp+56], xmm0        ; cos
    movsd xmm0, [rsp+32]
    call math_sin
    movapd xmm3, xmm0           ; sin
    movsd xmm2, [rsp+56]
    movsd xmm4, [rsp+40]
    movsd xmm5, [rsp+48]
    lea rax, [a0]
    cvtss2sd xmm0, [rax+rbx*8]  ; re
    cvtss2sd xmm1, [rax+rbx*8+4]
    ; (re + i im)(cos - i sin) = re cos + im sin + i (im cos - re sin)
    movapd xmm8, xmm0
    mulsd xmm8, xmm2
    movapd xmm9, xmm1
    mulsd xmm9, xmm3
    addsd xmm8, xmm9
    addsd xmm4, xmm8
    movapd xmm8, xmm1
    mulsd xmm8, xmm2
    movapd xmm9, xmm0
    mulsd xmm9, xmm3
    subsd xmm8, xmm9
    addsd xmm5, xmm8
    inc ebx
    cmp ebx, N1
    jb .n
    lea rax, [a]
    cvtss2sd xmm0, [rax+r12*8]
    subsd xmm0, xmm4
    andpd xmm0, [c_abs]
    maxsd xmm6, xmm0
    cvtss2sd xmm0, [rax+r12*8+4]
    subsd xmm0, xmm5
    andpd xmm0, [c_abs]
    maxsd xmm6, xmm0
    andpd xmm4, [c_abs]
    maxsd xmm7, xmm4
    inc r12d
    cmp r12d, N1
    jb .k
    divsd xmm6, xmm7
    movapd xmm0, xmm6
    close_to 0.0, 1e-5, "1024 points against a plain f64 dft, max diff over the largest"

    ; 2^20 there and back
    mov ecx, (1 << LBIG) * 8
    call mem_alloc
    mov [big], rax
    mov ecx, (1 << LBIG) * 8
    call mem_alloc
    mov [big0], rax
    xor ebx, ebx
.r2:
    call rnorm
    mov rax, [big]
    movss [rax+rbx*4], xmm0
    mov rax, [big0]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, 2 << LBIG
    jb .r2
    call time_now
    mov [rsp+32], rax
    mov rcx, [big]
    mov edx, LBIG
    xor r8d, r8d
    call fft
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    mov rcx, [big]
    mov edx, LBIG
    mov r8d, 1
    call fft
    mov eax, 1 << LBIG
    cvtsi2ss xmm2, eax
    xorpd xmm6, xmm6
    xor ebx, ebx
.c2:
    mov rax, [big]
    movss xmm0, [rax+rbx*4]
    divss xmm0, xmm2
    mov rax, [big0]
    subss xmm0, [rax+rbx*4]
    cvtss2sd xmm0, xmm0
    andpd xmm0, [c_abs]
    maxsd xmm6, xmm0
    inc ebx
    cmp ebx, 2 << LBIG
    jb .c2
    movapd xmm0, xmm6
    close_to 0.0, 1e-5, "2^20 points forward and back, max diff"
    say "  2^20 point fft on one thread: "
    movsd xmm0, [rsp+40]
    mulsd xmm0, [c_ms]
    mov edx, 1
    call print_fixed
    say " ms", 13, 10

    ; convolution
    xor ebx, ebx
.rx:
    call rnorm
    lea rax, [x]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NX
    jb .rx
    xor ebx, ebx
.rh:
    call rnorm
    lea rax, [h]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NH
    jb .rh
    lea rax, [scr]
    mov [rsp+32], rax
    mov qword [rsp+40], LCONV
    lea rax, [y]
    mov [rsp+48], rax
    lea rcx, [x]
    mov edx, NX
    lea r8, [h]
    mov r9d, NH
    call fft_conv
    xorpd xmm6, xmm6
    xorpd xmm7, xmm7
    xor ebx, ebx                ; t
.ct:
    xorpd xmm4, xmm4
    xor ecx, ecx                ; j
.cj:
    cmp ecx, NH
    jae .cjd
    cmp ecx, ebx
    ja .cjd
    mov eax, ebx
    sub eax, ecx
    lea rdx, [h]
    cvtss2sd xmm0, [rdx+rcx*4]
    lea rdx, [x]
    cvtss2sd xmm1, [rdx+rax*4]
    mulsd xmm0, xmm1
    addsd xmm4, xmm0
    inc ecx
    jmp .cj
.cjd:
    lea rax, [y]
    cvtss2sd xmm0, [rax+rbx*4]
    subsd xmm0, xmm4
    andpd xmm0, [c_abs]
    maxsd xmm6, xmm0
    andpd xmm4, [c_abs]
    maxsd xmm7, xmm4
    inc ebx
    cmp ebx, NX
    jb .ct
    divsd xmm6, xmm7
    movapd xmm0, xmm6
    close_to 0.0, 1e-5, "fft_conv (x + i h in one fft) against direct convolution"
    jmp t_done
