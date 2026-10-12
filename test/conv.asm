; stage 4 test: the conformer conv module's own kernels (glu, silu, the causal
; depthwise conv). a plain f64 version on the cpu, its gradient checked by finite
; differences, then the gpu kernels against it, in f32 and in bf16. the chain is
;   h -> glu -> g -> conv -> y -> silu -> s,   loss = sum r * s   (r random)
; uses: gpu\cuda gpu\kernels
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "gpu/cuda.inc"

extern ptx_conv, ptx_conv_end, ptx_ops, ptx_ops_end, ptx_basic, ptx_basic_end

NB      equ 3                   ; utterances
TT      equ 20                  ; frames each
C       equ 40                  ; channels: not a multiple of 32, so the edges get used
K       equ 9                   ; taps
M       equ NB * TT
NW      equ (K + 1) * C         ; the taps and the bias
NCH      equ 4                   ; row chunks for the weight gradient
; the big one, for speed: 32 utterances of 30 s at 80 ms frames, 512 channels
BIGT      equ 375
BM      equ 32 * BIGT
BC      equ 512
BCH     equ 64

section .rdata
k_glu    db "glu", 0
k_glub   db "glu_bwd", 0
k_silu   db "silu", 0
k_silub  db "silu_bwd", 0
k_conv   db "dwconv", 0
k_convx  db "dwconv_bwdx", 0
k_convw  db "dwconv_dw", 0
k_colsum db "colsum", 0
k_f2bf   db "f2bf", 0
align 8
c_one   dq 1.0
c_half  dq 0.5
c_eps   dq 1e-5
c_hs    dq 1.5
c_ws    dq 0.3
c_nan   dq 0x7ff8000000000000
align 16
c_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff

section .bss
alignb 8
kl      resb KL_SIZE
f_glu   resq 1
f_glub  resq 1
f_silu  resq 1
f_silub resq 1
f_conv  resq 1
f_convx resq 1
f_convw resq 1
f_colsum resq 1
f_f2bf  resq 1
rng     resb RNG_SIZE
; the shape the run_ helpers launch with
s_m     resd 1                  ; rows
s_t     resd 1                  ; frames an utterance
s_c     resd 1                  ; channels
s_bf    resd 1
s_ch    resd 1                  ; row chunks for the weight gradient
alignb 8
; the cpu's, f64
h64     resq M * 2 * C
w64     resq NW
r64     resq M * C
g64     resq M * C
y64     resq M * C
s64     resq M * C
dy64    resq M * C
dg64    resq M * C
dh64    resq M * 2 * C
dw64    resq NW
; what the gpu gets, and where its results come back
h32     resd M * 2 * C
w32     resd NW
r32     resd M * C
back    resd M * 2 * C
hb      resw M * 2 * C
alignb 8
d_h     resq 1
d_hb    resq 1
d_w     resq 1
d_r     resq 1
d_g     resq 1
d_y     resq 1
d_yb    resq 1
d_s     resq 1
d_dy    resq 1
d_dg    resq 1
d_dh    resq 1
d_part  resq 1
d_dw    resq 1

section .text

; rcx = module, the function goes in [rdx]
%macro getf 3
    mov rcx, %1
    lea rdx, [%2]
    call gpu_func
    mov [%3], rax
%endmacro

; ---- the cpu version, f64, straight from the definitions

; xmm0 = x. xmm0 = 1 / (1 + e^-x)
sig:
    sub rsp, 40
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
    call math_exp
    addsd xmm0, [c_one]
    movsd xmm1, [c_one]
    divsd xmm1, xmm0
    movapd xmm0, xmm1
    add rsp, 40
    ret

; the forward pass from h64 and w64: g, y and s. xmm0 = the loss, sum r * s
cpu_fwd:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    ; g = a * sigmoid(b)
    xor ebx, ebx
.glu:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    imul eax, eax, 2 * C
    add eax, edx
    mov esi, eax                ; a's index, b is C further
    lea rcx, [h64]
    movsd xmm0, [rcx+rsi*8+C*8]
    call sig
    lea rcx, [h64]
    mulsd xmm0, [rcx+rsi*8]
    lea rcx, [g64]
    movsd [rcx+rbx*8], xmm0
    inc ebx
    cmp ebx, M * C
    jb .glu
    ; y[t] = bias + sum_i w[i] g[t-K+1+i], frames before 0 left out
    xor ebx, ebx
.conv:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    mov r12d, eax               ; m
    mov r13d, edx               ; c
    xor edx, edx
    mov ecx, TT
    div ecx                     ; edx = t
    lea r8, [w64]
    movsd xmm0, [r8+r13*8+K*C*8]
    xor edi, edi
.tap:
    lea eax, [rdx+rdi-(K-1)]
    test eax, eax
    js .skip
    lea eax, [r12+rdi-(K-1)]
    imul eax, eax, C
    add eax, r13d
    lea rcx, [g64]
    movsd xmm1, [rcx+rax*8]
    imul eax, edi, C
    add eax, r13d
    mulsd xmm1, [r8+rax*8]
    addsd xmm0, xmm1
.skip:
    inc edi
    cmp edi, K
    jb .tap
    lea rcx, [y64]
    movsd [rcx+rbx*8], xmm0
    inc ebx
    cmp ebx, M * C
    jb .conv
    ; s = y * sigmoid(y), and the loss
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    xor ebx, ebx
.silu:
    lea rcx, [y64]
    movsd xmm0, [rcx+rbx*8]
    call sig
    lea rcx, [y64]
    mulsd xmm0, [rcx+rbx*8]
    lea rcx, [s64]
    movsd [rcx+rbx*8], xmm0
    lea rcx, [r64]
    mulsd xmm0, [rcx+rbx*8]
    addsd xmm0, [rsp+32]
    movsd [rsp+32], xmm0
    inc ebx
    cmp ebx, M * C
    jb .silu
    movsd xmm0, [rsp+32]
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the backward pass, after cpu_fwd: dy (at the conv's output), dg, dh and dw
cpu_bwd:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    ; dy = r s (1 + y (1 - s)), s = sigmoid(y)
    xor ebx, ebx
.dy:
    lea rcx, [y64]
    movsd xmm0, [rcx+rbx*8]
    call sig
    movsd xmm1, [c_one]
    subsd xmm1, xmm0
    lea rcx, [y64]
    mulsd xmm1, [rcx+rbx*8]
    addsd xmm1, [c_one]
    mulsd xmm0, xmm1
    lea rcx, [r64]
    mulsd xmm0, [rcx+rbx*8]
    lea rcx, [dy64]
    movsd [rcx+rbx*8], xmm0
    inc ebx
    cmp ebx, M * C
    jb .dy
    ; dg[t] = sum_i w[i] dy[t+K-1-i], the outputs inside the utterance
    xor ebx, ebx
.dg:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    mov r12d, eax
    mov r13d, edx
    xor edx, edx
    mov ecx, TT
    div ecx                     ; edx = t
    xorpd xmm0, xmm0
    xor edi, edi
.dgt:
    lea eax, [rdx+K-1]
    sub eax, edi
    cmp eax, TT
    jae .dgs
    lea eax, [r12+K-1]
    sub eax, edi
    imul eax, eax, C
    add eax, r13d
    lea rcx, [dy64]
    movsd xmm1, [rcx+rax*8]
    imul eax, edi, C
    add eax, r13d
    lea rcx, [w64]
    mulsd xmm1, [rcx+rax*8]
    addsd xmm0, xmm1
.dgs:
    inc edi
    cmp edi, K
    jb .dgt
    lea rcx, [dg64]
    movsd [rcx+rbx*8], xmm0
    inc ebx
    cmp ebx, M * C
    jb .dg
    ; dw[i][c] = sum over rows of dy g[t-K+1+i], the bias's is the sum of dy
    xor ebx, ebx
.dw:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    mov edi, eax                ; i
    mov r13d, edx               ; c
    xorpd xmm0, xmm0
    xor r12d, r12d
.dwm:
    imul eax, r12d, C
    add eax, r13d
    lea rcx, [dy64]
    movsd xmm1, [rcx+rax*8]
    cmp edi, K
    je .dwadd
    mov eax, r12d
    xor edx, edx
    mov ecx, TT
    div ecx
    lea eax, [rdx+rdi-(K-1)]
    test eax, eax
    js .dwn
    lea eax, [r12+rdi-(K-1)]
    imul eax, eax, C
    add eax, r13d
    lea rcx, [g64]
    mulsd xmm1, [rcx+rax*8]
.dwadd:
    addsd xmm0, xmm1
.dwn:
    inc r12d
    cmp r12d, M
    jb .dwm
    lea rcx, [dw64]
    movsd [rcx+rbx*8], xmm0
    inc ebx
    cmp ebx, NW
    jb .dw
    ; da = dg s, db = dg a s (1 - s), s = sigmoid(b)
    xor ebx, ebx
.dh:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    imul eax, eax, 2 * C
    add eax, edx
    mov esi, eax
    lea rcx, [h64]
    movsd xmm0, [rcx+rsi*8+C*8]
    call sig
    lea rcx, [dg64]
    movsd xmm1, [rcx+rbx*8]
    mulsd xmm1, xmm0
    lea rcx, [dh64]
    movsd [rcx+rsi*8], xmm1
    movsd xmm2, [c_one]
    subsd xmm2, xmm0
    mulsd xmm1, xmm2
    lea rcx, [h64]
    mulsd xmm1, [rcx+rsi*8]
    lea rcx, [dh64]
    movsd [rcx+rsi*8+C*8], xmm1
    inc ebx
    cmp ebx, M * C
    jb .dh
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = f64 array, edx = index. xmm0 = (L(x + eps) - L(x - eps)) / 2 eps
fd:
    push rbx
    push rsi
    sub rsp, 56
    mov rbx, rcx
    mov esi, edx
    movsd xmm0, [rbx+rsi*8]
    movsd [rsp+32], xmm0
    addsd xmm0, [c_eps]
    movsd [rbx+rsi*8], xmm0
    call cpu_fwd
    movsd [rsp+40], xmm0
    movsd xmm0, [rsp+32]
    subsd xmm0, [c_eps]
    movsd [rbx+rsi*8], xmm0
    call cpu_fwd
    movsd xmm1, [rsp+40]
    subsd xmm1, xmm0
    mulsd xmm1, [c_half]
    divsd xmm1, [c_eps]
    movsd xmm0, [rsp+32]
    movsd [rbx+rsi*8], xmm0
    movapd xmm0, xmm1
    add rsp, 56
    pop rsi
    pop rbx
    ret

; xmm0 = a new difference, rcx = the running max. a nan sticks (maxsd alone would
; drop it again on the next number)
keepmax:
    movsd xmm1, [rcx]
    ucomisd xmm1, xmm1
    jp .done
    ucomisd xmm0, xmm0
    jp .put
    maxsd xmm0, xmm1
.put:
    movsd [rcx], xmm0
.done:
    ret

; rcx = what the gpu gave, edx = 1 if that's bf16 (else f32), r8 = the cpu's f64,
; r9d = count. xmm0 = max |gpu - cpu| / max |cpu|, nan if any were nan
cmprel:
    xorpd xmm2, xmm2
    xorpd xmm3, xmm3
    xor r11d, r11d
    xor eax, eax
.l:
    test edx, edx
    jnz .b
    cvtss2sd xmm0, [rcx+rax*4]
    jmp .c
.b:
    movzx r10d, word [rcx+rax*2]
    shl r10d, 16
    movd xmm0, r10d
    cvtss2sd xmm0, xmm0
.c:
    movsd xmm1, [r8+rax*8]
    subsd xmm0, xmm1
    ucomisd xmm0, xmm0
    jnp .n
    inc r11d
.n:
    andpd xmm0, [c_abs]
    andpd xmm1, [c_abs]
    maxsd xmm2, xmm0
    maxsd xmm3, xmm1
    inc eax
    cmp eax, r9d
    jb .l
    divsd xmm2, xmm3
    movapd xmm0, xmm2
    test r11d, r11d
    jz .done
    movsd xmm0, [c_nan]
.done:
    ret

; rcx = device buffer, rdx = the cpu's f64, r8d = count, r9d = 1 if bf16.
; brings it back and compares: xmm0 as cmprel
against:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, rdx
    mov esi, r8d
    mov edi, r9d
    mov rdx, rcx
    lea rcx, [back]
    mov eax, 4
    sub eax, edi
    sub eax, edi
    imul r8d, eax
    call gpu_down
    lea rcx, [back]
    mov edx, edi
    mov r8, rbx
    mov r9d, esi
    call cmprel
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; r8d = f32 bits -> r8w = bf16 bits, round to nearest even (no nans in here)
bf16:
    mov r10d, r8d
    shr r10d, 16
    and r10d, 1
    add r8d, 0x7fff
    add r8d, r10d
    shr r8d, 16
    ret

; ---- the gpu side. the run_ helpers launch with the shape in s_*

; rbx = kl with the args in. rax = the function, ecx = how many threads, 256 a block
go_flat:
    sub rsp, 40
    mov [rbx+KL_FUNC], rax
    lea eax, [rcx+255]
    shr eax, 8
    mov dword [rbx+KL_GX], eax
    mov dword [rbx+KL_GY], 1
    mov dword [rbx+KL_GZ], 1
    mov dword [rbx+KL_BX], 256
    mov dword [rbx+KL_BY], 1
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    add rsp, 40
    ret

; rcx = h, rdx = g
run_glu:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov eax, [s_m]
    mov [rbx+KL_ARGS+16], rax
    mov eax, [s_c]
    mov [rbx+KL_ARGS+24], rax
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+32], rax
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov rax, [f_glu]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = h, rdx = dg, r8 = dh
run_glub:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov eax, [s_m]
    mov [rbx+KL_ARGS+24], rax
    mov eax, [s_c]
    mov [rbx+KL_ARGS+32], rax
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+40], rax
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov rax, [f_glub]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = x, rdx = y
run_silu:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov [rbx+KL_ARGS+16], rcx
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [f_silu]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = x, rdx = dy, r8 = dx
run_silub:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov [rbx+KL_ARGS+24], rcx
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+32], rax
    mov rax, [f_silub]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; the conv's args: rcx, rdx, r8 = the three pointers, then m, t, c, K, bf
conv_args:
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov eax, [s_m]
    mov [rbx+KL_ARGS+24], rax
    mov eax, [s_t]
    mov [rbx+KL_ARGS+32], rax
    mov eax, [s_c]
    mov [rbx+KL_ARGS+40], rax
    mov qword [rbx+KL_ARGS+48], K
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+56], rax
    ret

; rcx = x, rdx = w, r8 = y
run_conv:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    call conv_args
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov rax, [f_conv]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = dy, rdx = w, r8 = dx
run_convx:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    call conv_args
    mov ecx, [s_m]
    imul ecx, [s_c]
    mov rax, [f_convx]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = dy, rdx = x, r8 = partials, r9 = dw (added to)
run_convw:
    push rbx
    push rsi
    sub rsp, 40
    lea rbx, [kl]
    mov rsi, r9
    mov [rsp+32], r8
    call conv_args
    mov rax, [f_convw]
    mov [rbx+KL_FUNC], rax
    mov eax, [s_c]
    add eax, 31
    shr eax, 5
    mov dword [rbx+KL_GX], eax
    mov eax, [s_ch]
    mov dword [rbx+KL_GY], eax
    mov dword [rbx+KL_GZ], K + 1
    mov dword [rbx+KL_BX], 32
    mov dword [rbx+KL_BY], 8
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    mov rax, [rsp+32]
    mov [rbx+KL_ARGS], rax
    mov [rbx+KL_ARGS+8], rsi
    mov ecx, [s_c]
    imul ecx, ecx, K + 1
    mov [rbx+KL_ARGS+16], rcx
    mov eax, [s_ch]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [f_colsum]
    call go_flat
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = dst (bf16), rdx = src (f32), r8d = count
run_f2bf:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov ecx, r8d
    mov rax, [f_f2bf]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = device buffer, edx = bytes. zeroes it
zero:
    sub rsp, 40
    mov r8d, edx
    shr r8d, 2
    xor edx, edx
    CU cuMemsetD32
    add rsp, 40
    ret

; ecx = bytes. rax = device buffer, zeroed
galloc:
    push rbx
    push rsi
    sub rsp, 40
    mov esi, ecx
    call gpu_alloc
    mov rbx, rax
    mov rcx, rax
    mov edx, esi
    call zero
    mov rax, rbx
    add rsp, 40
    pop rsi
    pop rbx
    ret

; the whole chain on the gpu, forward and back, with whatever s_bf says. in bf16
; the conv's f32 output gets rounded (f2bf) before silu, as it would come out of
; the norm in the model
gpu_chain:
    sub rsp, 40
    cmp dword [s_bf], 0
    jne .bf
    mov rcx, [d_h]
    mov rdx, [d_g]
    call run_glu
    mov rcx, [d_g]
    mov rdx, [d_w]
    mov r8, [d_y]
    call run_conv
    mov rax, [d_y]
    mov [d_yb], rax             ; f32 all the way: silu reads y itself
    jmp .silu
.bf:
    mov rcx, [d_hb]
    mov rdx, [d_g]
    call run_glu
    mov rcx, [d_g]
    mov rdx, [d_w]
    mov r8, [d_y]
    call run_conv
    mov rcx, [d_yb]
    mov rdx, [d_y]
    mov r8d, [s_m]
    imul r8d, [s_c]
    call run_f2bf
.silu:
    mov rcx, [d_yb]
    mov rdx, [d_s]
    call run_silu
    mov rcx, [d_yb]
    mov rdx, [d_r]
    mov r8, [d_dy]
    call run_silub
    mov rcx, [d_dy]
    mov rdx, [d_w]
    mov r8, [d_dg]
    call run_convx
    mov rcx, [d_dw]
    imul edx, [s_c], (K + 1) * 4
    call zero
    mov rcx, [d_dy]
    mov rdx, [d_g]
    mov r8, [d_part]
    mov r9, [d_dw]
    call run_convw
    mov rcx, [d_h]
    cmp dword [s_bf], 0
    je .h
    mov rcx, [d_hb]
.h:
    mov rdx, [d_dg]
    mov r8, [d_dh]
    call run_glub
    call gpu_sync
    add rsp, 40
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    call lib_init
    call gpu_init
    lea rcx, [ptx_conv]
    lea rdx, [ptx_conv_end]
    sub rdx, rcx
    call gpu_module
    mov r12, rax
    getf r12, k_glu, f_glu
    getf r12, k_glub, f_glub
    getf r12, k_silu, f_silu
    getf r12, k_silub, f_silub
    getf r12, k_conv, f_conv
    getf r12, k_convx, f_convx
    getf r12, k_convw, f_convw
    lea rcx, [ptx_ops]
    lea rdx, [ptx_ops_end]
    sub rdx, rcx
    call gpu_module
    getf rax, k_colsum, f_colsum
    lea rcx, [ptx_basic]
    lea rdx, [ptx_basic_end]
    sub rdx, rcx
    call gpu_module
    getf rax, k_f2bf, f_f2bf

    ; h: normal x 1.5, rounded to bf16 so the cpu, the f32 run and the bf16 run
    ; all start from the same numbers
    lea rcx, [rng]
    mov edx, 5
    call rng_seed
    xor ebx, ebx
.gh:
    lea rcx, [rng]
    call rng_normal
    mulsd xmm0, [c_hs]
    cvtsd2ss xmm0, xmm0
    movd r8d, xmm0
    call bf16
    lea rax, [hb]
    mov [rax+rbx*2], r8w
    shl r8d, 16
    lea rax, [h32]
    mov [rax+rbx*4], r8d
    movd xmm0, r8d
    cvtss2sd xmm0, xmm0
    lea rax, [h64]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, M * 2 * C
    jb .gh
    ; w: normal x 0.3, the bias too. r: normal. both exact in f32
    xor ebx, ebx
.gw:
    lea rcx, [rng]
    call rng_normal
    mulsd xmm0, [c_ws]
    cvtsd2ss xmm0, xmm0
    lea rax, [w32]
    movss [rax+rbx*4], xmm0
    cvtss2sd xmm0, xmm0
    lea rax, [w64]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, NW
    jb .gw
    xor ebx, ebx
.gr:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    lea rax, [r32]
    movss [rax+rbx*4], xmm0
    cvtss2sd xmm0, xmm0
    lea rax, [r64]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, M * C
    jb .gr

    say "the cpu version, f64", 13, 10
    call cpu_fwd
    call cpu_bwd
    ; finite differences: 30 entries of h, and 20 weights, every tap and the bias twice
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    xor r13d, r13d
.fdh:
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    mov ecx, M * 2 * C
    div rcx
    mov esi, edx
    lea rcx, [h64]
    call fd
    lea rax, [dh64]
    subsd xmm0, [rax+rsi*8]
    andpd xmm0, [c_abs]
    lea rcx, [rsp+32]
    call keepmax
    inc r13d
    cmp r13d, 30
    jb .fdh
    xor r13d, r13d
.fdw:
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    mov ecx, C
    div rcx
    mov esi, edx
    mov eax, r13d
    xor edx, edx
    mov ecx, K + 1
    div ecx
    imul edx, edx, C
    add esi, edx                ; tap r13 % (K+1), a random channel
    mov edx, esi
    lea rcx, [w64]
    call fd
    lea rax, [dw64]
    subsd xmm0, [rax+rsi*8]
    andpd xmm0, [c_abs]
    lea rcx, [rsp+32]
    call keepmax
    inc r13d
    cmp r13d, 20
    jb .fdw
    call cpu_fwd                ; g, y and s as they were
    movsd xmm0, [rsp+32]
    close_to 0.0, 1e-6, "  formula vs finite differences, 30 inputs and 20 weights, max diff"

    say "the gpu kernels in f32 vs the cpu", 13, 10
    mov ecx, M * 2 * C * 4
    call galloc
    mov [d_h], rax
    mov ecx, M * 2 * C * 2
    call galloc
    mov [d_hb], rax
    mov ecx, NW * 4
    call galloc
    mov [d_w], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_r], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_g], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_y], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_s], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_dy], rax
    mov ecx, M * C * 4
    call galloc
    mov [d_dg], rax
    mov ecx, M * 2 * C * 4
    call galloc
    mov [d_dh], rax
    mov ecx, NCH * NW * 4
    call galloc
    mov [d_part], rax
    mov ecx, NW * 4
    call galloc
    mov [d_dw], rax
    mov rcx, [d_h]
    lea rdx, [h32]
    mov r8d, M * 2 * C * 4
    call gpu_up
    mov rcx, [d_hb]
    lea rdx, [hb]
    mov r8d, M * 2 * C * 2
    call gpu_up
    mov rcx, [d_w]
    lea rdx, [w32]
    mov r8d, NW * 4
    call gpu_up
    mov rcx, [d_r]
    lea rdx, [r32]
    mov r8d, M * C * 4
    call gpu_up
    mov dword [s_m], M
    mov dword [s_t], TT
    mov dword [s_c], C
    mov dword [s_bf], 0
    mov dword [s_ch], NCH
    call gpu_chain
    mov rcx, [d_g]
    lea rdx, [g64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  glu, max diff over the largest"
    mov rcx, [d_y]
    lea rdx, [y64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  causal conv"
    mov rcx, [d_s]
    lea rdx, [s64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  silu"
    mov rcx, [d_dy]
    lea rdx, [dy64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  silu backward"
    mov rcx, [d_dg]
    lea rdx, [dg64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv backward, inputs"
    mov rcx, [d_dw]
    lea rdx, [dw64]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv backward, taps and bias"
    mov rcx, [d_dh]
    lea rdx, [dh64]
    mov r8d, M * 2 * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  glu backward"

    say "the same in bf16 (rounding on every T tensor)", 13, 10
    mov ecx, M * C * 2
    call galloc
    mov [d_yb], rax
    mov dword [s_bf], 1
    call gpu_chain
    mov rcx, [d_y]
    lea rdx, [y64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 1e-2, "  causal conv"
    mov rcx, [d_dy]
    lea rdx, [dy64]
    mov r8d, M * C
    xor r9d, r9d
    call against
    close_to 0.0, 2e-2, "  silu backward"
    mov rcx, [d_dw]
    lea rdx, [dw64]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 2e-2, "  conv backward, taps and bias"
    mov rcx, [d_dh]
    lea rdx, [dh64]
    mov r8d, M * 2 * C
    mov r9d, 1
    call against
    close_to 0.0, 2e-2, "  glu backward"

    say "speed", 13, 10
    call big
    jmp t_done

; the real size: 32 x 375 frames, 512 channels, bf16. timed 20 times each way
big:
    push rbx
    sub rsp, 48
    mov ecx, BM * 2 * BC * 2
    call galloc
    mov [d_hb], rax
    mov ecx, (K + 1) * BC * 4
    call galloc
    mov [d_w], rax
    mov ecx, BM * BC * 4
    call galloc
    mov [d_r], rax
    mov ecx, BM * BC * 2
    call galloc
    mov [d_g], rax
    mov ecx, BM * BC * 4
    call galloc
    mov [d_y], rax
    mov ecx, BM * BC * 2
    call galloc
    mov [d_yb], rax
    mov ecx, BM * BC * 2
    call galloc
    mov [d_s], rax
    mov ecx, BM * BC * 4
    call galloc
    mov [d_dy], rax
    mov ecx, BM * BC * 2
    call galloc
    mov [d_dg], rax
    mov ecx, BM * 2 * BC * 2
    call galloc
    mov [d_dh], rax
    mov ecx, BCH * (K + 1) * BC * 4
    call galloc
    mov [d_part], rax
    mov ecx, (K + 1) * BC * 4
    call galloc
    mov [d_dw], rax
    mov dword [s_m], BM
    mov dword [s_t], BIGT
    mov dword [s_c], BC
    mov dword [s_bf], 1
    mov dword [s_ch], BCH
    call gpu_chain              ; warm up
    ; forward: glu, conv, silu
    xor ecx, ecx
    call gpu_tstart
    mov ebx, 20
.f:
    mov rcx, [d_hb]
    mov rdx, [d_g]
    call run_glu
    mov rcx, [d_g]
    mov rdx, [d_w]
    mov r8, [d_y]
    call run_conv
    mov rcx, [d_yb]
    mov rdx, [d_s]
    call run_silu
    dec ebx
    jnz .f
    xor ecx, ecx
    call gpu_tstop
    mov eax, 20
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+32], xmm0
    ; backward: silu, conv (inputs, then taps and bias), glu
    xor ecx, ecx
    call gpu_tstart
    mov ebx, 20
.b:
    mov rcx, [d_yb]
    mov rdx, [d_r]
    mov r8, [d_dy]
    call run_silub
    mov rcx, [d_dy]
    mov rdx, [d_w]
    mov r8, [d_dg]
    call run_convx
    mov rcx, [d_dy]
    mov rdx, [d_g]
    mov r8, [d_part]
    mov r9, [d_dw]
    call run_convw
    mov rcx, [d_hb]
    mov rdx, [d_dg]
    mov r8, [d_dh]
    call run_glub
    dec ebx
    jnz .b
    xor ecx, ecx
    call gpu_tstop
    mov eax, 20
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+40], xmm0
    say "  32 x 375 frames x 512 channels, bf16: forward "
    movsd xmm0, [rsp+32]
    mov edx, 3
    call print_fixed
    say " ms, backward "
    movsd xmm0, [rsp+40]
    mov edx, 3
    call print_fixed
    say " ms", 13, 10
    add rsp, 48
    pop rbx
    ret
