; stage 4 test: the subsampling front end's kernels (sconv, its backward for x and
; for the weights, relu). a plain f64 version on the cpu, its gradient checked by
; finite differences, then the gpu kernels against it, in f32 and in bf16. the chain:
;   mel x -> conv1 (full, relu) -> a1 -> depthwise conv2 -> b2 -> relu -> a2,
;   loss = sum r * a2   (r random)
; the cpu works the input gradient out by scattering (each output hands its share
; back), the gpu by gathering, so the two are written independently
; uses: gpu\cuda gpu\kernels
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "gpu/cuda.inc"
%include "test/gpucheck.inc"

extern ptx_sub, ptx_sub_end, ptx_ops, ptx_ops_end

NB      equ 2                   ; utterances
T0      equ 13                  ; frames, odd so the rounding up gets used
F0      equ 10                  ; bins
C       equ 40                  ; channels, not a multiple of 32
T1      equ (T0 + 1) / 2
F1      equ (F0 + 1) / 2
T2      equ (T1 + 1) / 2
F2      equ (F1 + 1) / 2
NX      equ NB * T0 * F0
N1      equ NB * T1 * F1 * C
N2      equ NB * T2 * F2 * C
NW      equ 10 * C              ; nine taps and the bias
NCH     equ 3                   ; output chunks for the weight gradients
; the real size: 32 utterances of 15 s, 80 bins, 256 channels
BB      equ 32
BT0     equ 1500
BF0     equ 80
BC      equ 256
BT1     equ (BT0 + 1) / 2
BF1     equ (BF0 + 1) / 2
BT2     equ (BT1 + 1) / 2
BF2     equ (BF1 + 1) / 2
BT3     equ (BT2 + 1) / 2
BF3     equ (BF2 + 1) / 2
BCH     equ 256

section .rdata
k_sconv  db "sconv", 0
k_sbx    db "sconv_bwdx", 0
k_sdw    db "sconv_dw", 0
k_relu   db "relu", 0
k_relub  db "relu_bwd", 0
k_colsum db "colsum", 0
align 8
c_half  dq 0.5
c_eps   dq 1e-5
c_ws    dq 0.3

section .bss
alignb 8
f_sconv resq 1
f_sbx   resq 1
f_sdw   resq 1
f_relu  resq 1
f_relub resq 1
f_colsum resq 1
rng     resb RNG_SIZE
; the launch shape: s_b utterances, s_t frames and s_f bins going in, s_c channels,
; s_fl the conv flags besides bf16, s_ch chunks
s_b     resd 1
s_t     resd 1
s_f     resd 1
s_c     resd 1
s_bf    resd 1
s_fl    resd 1
s_ch    resd 1
alignb 8
; the cpu conv's arguments
v_x     resq 1
v_w     resq 1
v_y     resq 1
v_dy    resq 1
v_dx    resq 1
v_dw    resq 1
v_t     resd 1
v_f     resd 1
v_fl    resd 1
v_to    resd 1
v_fo    resd 1
v_n     resd 1
alignb 8
; the cpu's, f64
x64     resq NX
w164    resq NW
w264    resq NW
r64     resq N2
a164    resq N1
b264    resq N2
a264    resq N2
db264   resq N2
da164   resq N1
dp164   resq N1
dw164   resq NW
dw264   resq NW
; what the gpu gets, and where its results come back
x32     resd NX
w132    resd NW
w232    resd NW
r32     resd N2
back    resd N1
rb      resw N2
alignb 8
d_x     resq 1
d_w1    resq 1
d_w2    resq 1
d_r     resq 1
d_rb    resq 1
d_a1    resq 1
d_b2    resq 1
d_a2    resq 1
d_db2   resq 1
d_da1   resq 1
d_dp1   resq 1
d_part  resq 1
d_dw1   resq 1
d_dw2   resq 1
; the big run's
d_a3    resq 1
d_da2   resq 1
d_da3   resq 1

section .text

; ---- the cpu version, f64, straight from the definition

; v_to, v_fo and v_n (outputs) from v_t and v_f
shape:
    mov eax, [v_t]
    inc eax
    shr eax, 1
    mov [v_to], eax
    mov ecx, [v_f]
    inc ecx
    shr ecx, 1
    mov [v_fo], ecx
    imul eax, ecx
    imul eax, eax, NB * C
    mov [v_n], eax
    ret

; output index ebx -> r12d = c, r13d = p, r14d = o, r15d = b
split:
    mov eax, ebx
    xor edx, edx
    mov ecx, C
    div ecx
    mov r12d, edx
    xor edx, edx
    div dword [v_fo]
    mov r13d, edx
    xor edx, edx
    div dword [v_to]
    mov r14d, edx
    mov r15d, eax
    ret

; esi = i, edi = j, with the output in r12-r15. CF set when that tap reads outside
; (padding), else eax = the x index it reads and ecx = the w index
tapat:
    lea eax, [r14*2+rsi-2]      ; t
    test eax, eax
    js .out
    mov r9d, eax
    lea eax, [r13*2+rdi-1]      ; f
    test eax, eax
    js .out
    cmp eax, [v_f]
    jge .out
    mov r10d, eax
    mov eax, r15d
    imul eax, [v_t]
    add eax, r9d
    imul eax, [v_f]
    add eax, r10d
    test dword [v_fl], 2
    jnz .one
    imul eax, eax, C
    add eax, r12d
.one:
    lea ecx, [rsi+rsi*2]
    add ecx, edi
    imul ecx, ecx, C
    add ecx, r12d
    clc
    ret
.out:
    stc
    ret

; y = act(bias + sum of taps), v_x, v_w -> v_y with v_t, v_f, v_fl
cpu_sconv:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    call shape
    xor ebx, ebx
.o:
    call split
    mov r8, [v_w]
    lea eax, [r12+9*C]
    movsd xmm0, [r8+rax*8]
    xor esi, esi
.i:
    xor edi, edi
.j:
    call tapat
    jc .nj
    mov r8, [v_x]
    movsd xmm1, [r8+rax*8]
    mov r8, [v_w]
    mulsd xmm1, [r8+rcx*8]
    addsd xmm0, xmm1
.nj:
    inc edi
    cmp edi, 3
    jb .j
    inc esi
    cmp esi, 3
    jb .i
    test dword [v_fl], 4
    jz .st
    xorpd xmm1, xmm1
    maxsd xmm0, xmm1
.st:
    mov r8, [v_y]
    movsd [r8+rbx*8], xmm0
    inc ebx
    cmp ebx, [v_n]
    jb .o
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; backward: v_dy (at the conv's output, before any relu) -> v_dw, and v_dx unless
; x is the one-channel input. every output hands its gradient to what it read
cpu_sconvb:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    call shape
    mov rdi, [v_dw]
    mov ecx, NW
    xor eax, eax
    rep stosq
    test dword [v_fl], 2
    jnz .nz
    mov eax, [v_t]
    imul eax, [v_f]
    imul eax, eax, NB * C
    mov ecx, eax
    mov rdi, [v_dx]
    xor eax, eax
    rep stosq
.nz:
    xor ebx, ebx
.o:
    call split
    mov r8, [v_dy]
    movsd xmm0, [r8+rbx*8]
    mov r8, [v_dw]
    lea eax, [r12+9*C]
    movsd xmm1, [r8+rax*8]
    addsd xmm1, xmm0
    movsd [r8+rax*8], xmm1
    xor esi, esi
.i:
    xor edi, edi
.j:
    call tapat
    jc .nj
    mov r11d, eax
    mov r8, [v_x]
    movsd xmm1, [r8+r11*8]
    mulsd xmm1, xmm0
    mov r8, [v_dw]
    addsd xmm1, [r8+rcx*8]
    movsd [r8+rcx*8], xmm1
    test dword [v_fl], 2
    jnz .nj
    mov r8, [v_w]
    movsd xmm1, [r8+rcx*8]
    mulsd xmm1, xmm0
    mov r8, [v_dx]
    addsd xmm1, [r8+r11*8]
    movsd [r8+r11*8], xmm1
.nj:
    inc edi
    cmp edi, 3
    jb .j
    inc esi
    cmp esi, 3
    jb .i
    inc ebx
    cmp ebx, [v_n]
    jb .o
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the chain forward from x64, w164, w264. xmm0 = the loss
cpu_fwd:
    push rbx
    sub rsp, 48
    lea rax, [x64]
    mov [v_x], rax
    lea rax, [w164]
    mov [v_w], rax
    lea rax, [a164]
    mov [v_y], rax
    mov dword [v_t], T0
    mov dword [v_f], F0
    mov dword [v_fl], 6
    call cpu_sconv
    lea rax, [a164]
    mov [v_x], rax
    lea rax, [w264]
    mov [v_w], rax
    lea rax, [b264]
    mov [v_y], rax
    mov dword [v_t], T1
    mov dword [v_f], F1
    mov dword [v_fl], 0
    call cpu_sconv
    xorpd xmm0, xmm0
    xor ebx, ebx
.l:
    lea rax, [b264]
    movsd xmm1, [rax+rbx*8]
    xorpd xmm2, xmm2
    maxsd xmm1, xmm2
    lea rax, [a264]
    movsd [rax+rbx*8], xmm1
    lea rax, [r64]
    mulsd xmm1, [rax+rbx*8]
    addsd xmm0, xmm1
    inc ebx
    cmp ebx, N2
    jb .l
    add rsp, 48
    pop rbx
    ret

; and back, after cpu_fwd: db2, dw2, da1, dp1, dw1
cpu_bwd:
    push rbx
    sub rsp, 48
    xor ebx, ebx
.r2:
    lea rax, [r64]
    movsd xmm1, [rax+rbx*8]
    lea rax, [b264]
    movsd xmm2, [rax+rbx*8]
    xorpd xmm3, xmm3
    comisd xmm2, xmm3
    ja .k2
    xorpd xmm1, xmm1
.k2:
    lea rax, [db264]
    movsd [rax+rbx*8], xmm1
    inc ebx
    cmp ebx, N2
    jb .r2
    lea rax, [db264]
    mov [v_dy], rax
    lea rax, [a164]
    mov [v_x], rax
    lea rax, [w264]
    mov [v_w], rax
    lea rax, [dw264]
    mov [v_dw], rax
    lea rax, [da164]
    mov [v_dx], rax
    mov dword [v_t], T1
    mov dword [v_f], F1
    mov dword [v_fl], 0
    call cpu_sconvb
    xor ebx, ebx
.r1:
    lea rax, [da164]
    movsd xmm1, [rax+rbx*8]
    lea rax, [a164]
    movsd xmm2, [rax+rbx*8]
    xorpd xmm3, xmm3
    comisd xmm2, xmm3
    ja .k1
    xorpd xmm1, xmm1
.k1:
    lea rax, [dp164]
    movsd [rax+rbx*8], xmm1
    inc ebx
    cmp ebx, N1
    jb .r1
    lea rax, [dp164]
    mov [v_dy], rax
    lea rax, [x64]
    mov [v_x], rax
    lea rax, [w164]
    mov [v_w], rax
    lea rax, [dw164]
    mov [v_dw], rax
    mov dword [v_t], T0
    mov dword [v_f], F0
    mov dword [v_fl], 2
    call cpu_sconvb
    add rsp, 48
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

; rcx = w (f64), rdx = its formula gradient, [rsp+32 of the caller] gets the max.
; 20 weights: every tap and the bias twice, random channels
fdw:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov r12, rcx
    mov r13, rdx
    mov rdi, r8
    xor ebx, ebx
.l:
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    mov ecx, C
    div rcx
    mov esi, edx
    mov eax, ebx
    xor edx, edx
    mov ecx, 10
    div ecx
    imul edx, edx, C
    add esi, edx
    mov rcx, r12
    mov edx, esi
    call fd
    subsd xmm0, [r13+rsi*8]
    andpd xmm0, [t_abs]
    mov rcx, rdi
    call keepmax
    inc ebx
    cmp ebx, 20
    jb .l
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ---- the gpu side. the run_ helpers launch with the shape in s_*

; rcx, rdx, r8 = the three pointers. the rest of a conv's arguments
conv_args:
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov eax, [s_b]
    mov [rbx+KL_ARGS+24], rax
    mov eax, [s_t]
    mov [rbx+KL_ARGS+32], rax
    mov eax, [s_f]
    mov [rbx+KL_ARGS+40], rax
    mov eax, [s_c]
    mov [rbx+KL_ARGS+48], rax
    mov eax, [s_fl]
    or eax, [s_bf]
    mov [rbx+KL_ARGS+56], rax
    ret

; rcx = x, rdx = w, r8 = y
run_sconv:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    call conv_args
    mov ecx, [s_t]
    inc ecx
    shr ecx, 1
    mov eax, [s_f]
    inc eax
    shr eax, 1
    imul ecx, eax
    imul ecx, [s_b]
    imul ecx, [s_c]
    mov rax, [f_sconv]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = dy, rdx = w, r8 = dx
run_sbx:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    call conv_args
    mov ecx, [s_t]
    imul ecx, [s_f]
    imul ecx, [s_b]
    imul ecx, [s_c]
    mov rax, [f_sbx]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = dy, rdx = x, r8 = partials, r9 = dw (added to)
run_sdw:
    push rbx
    push rsi
    sub rsp, 40
    lea rbx, [kl]
    mov rsi, r9
    mov [rsp+32], r8
    call conv_args
    mov rax, [f_sdw]
    mov [rbx+KL_FUNC], rax
    mov eax, [s_c]
    add eax, 31
    shr eax, 5
    mov dword [rbx+KL_GX], eax
    mov eax, [s_ch]
    mov dword [rbx+KL_GY], eax
    mov dword [rbx+KL_GZ], 1
    mov dword [rbx+KL_BX], 32
    mov dword [rbx+KL_BY], 8
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    mov rax, [rsp+32]
    mov [rbx+KL_ARGS], rax
    mov [rbx+KL_ARGS+8], rsi
    mov ecx, [s_c]
    imul ecx, ecx, 10
    mov [rbx+KL_ARGS+16], rcx
    mov eax, [s_ch]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [f_colsum]
    call go_flat
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = x, rdx = y, r8d = count
run_relu:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+24], rax
    mov ecx, r8d
    mov rax, [f_relu]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; rcx = y, rdx = dy, r8 = dx, r9d = count
run_relub:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    mov [rbx+KL_ARGS+24], r9
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+32], rax
    mov ecx, r9d
    mov rax, [f_relub]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; the chain on the gpu, forward and back, with whatever s_bf says
gpu_chain:
    sub rsp, 40
    mov dword [s_t], T0
    mov dword [s_f], F0
    mov dword [s_fl], 6
    mov rcx, [d_x]
    mov rdx, [d_w1]
    mov r8, [d_a1]
    call run_sconv
    mov dword [s_t], T1
    mov dword [s_f], F1
    mov dword [s_fl], 0
    mov rcx, [d_a1]
    mov rdx, [d_w2]
    mov r8, [d_b2]
    call run_sconv
    mov rcx, [d_b2]
    mov rdx, [d_a2]
    mov r8d, N2
    call run_relu
    mov rcx, [d_a2]
    mov rdx, [d_r]
    cmp dword [s_bf], 0
    je .r
    mov rdx, [d_rb]
.r:
    mov r8, [d_db2]
    mov r9d, N2
    call run_relub
    mov rcx, [d_db2]
    mov rdx, [d_w2]
    mov r8, [d_da1]
    call run_sbx
    mov rcx, [d_dw2]
    mov edx, NW * 4
    call zero
    mov rcx, [d_db2]
    mov rdx, [d_a1]
    mov r8, [d_part]
    mov r9, [d_dw2]
    call run_sdw
    mov rcx, [d_a1]
    mov rdx, [d_da1]
    mov r8, [d_dp1]
    mov r9d, N1
    call run_relub
    mov dword [s_t], T0
    mov dword [s_f], F0
    mov dword [s_fl], 2
    mov rcx, [d_dw1]
    mov edx, NW * 4
    call zero
    mov rcx, [d_dp1]
    mov rdx, [d_x]
    mov r8, [d_part]
    mov r9, [d_dw1]
    call run_sdw
    call gpu_sync
    add rsp, 40
    ret

; rcx = rng-filled f32 out, rdx = f64 out, r8d = count, xmm0 = scale. normal numbers,
; exact in f32
fill:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    mov rsi, rcx
    mov rdi, rdx
    mov r12d, r8d
    movsd [rsp+32], xmm0
    xor ebx, ebx
.l:
    lea rcx, [rng]
    call rng_normal
    mulsd xmm0, [rsp+32]
    cvtsd2ss xmm0, xmm0
    movss [rsi+rbx*4], xmm0
    cvtss2sd xmm0, xmm0
    movsd [rdi+rbx*8], xmm0
    inc ebx
    cmp ebx, r12d
    jb .l
    add rsp, 56
    pop r12
    pop rdi
    pop rsi
    pop rbx
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
    lea rcx, [ptx_sub]
    lea rdx, [ptx_sub_end]
    sub rdx, rcx
    call gpu_module
    mov r12, rax
    getf r12, k_sconv, f_sconv
    getf r12, k_sbx, f_sbx
    getf r12, k_sdw, f_sdw
    getf r12, k_relu, f_relu
    getf r12, k_relub, f_relub
    lea rcx, [ptx_ops]
    lea rdx, [ptx_ops_end]
    sub rdx, rcx
    call gpu_module
    getf rax, k_colsum, f_colsum

    ; x and both convs' weights: normal (x 0.3 for the weights), exact in f32.
    ; r: normal, rounded to bf16 so the f32 and bf16 runs both get it exactly
    lea rcx, [rng]
    mov edx, 7
    call rng_seed
    lea rcx, [x32]
    lea rdx, [x64]
    mov r8d, NX
    movsd xmm0, [c_one]
    call fill
    lea rcx, [w132]
    lea rdx, [w164]
    mov r8d, NW
    movsd xmm0, [c_ws]
    call fill
    lea rcx, [w232]
    lea rdx, [w264]
    mov r8d, NW
    movsd xmm0, [c_ws]
    call fill
    xor ebx, ebx
.gr:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    movd r8d, xmm0
    call bf16
    lea rax, [rb]
    mov [rax+rbx*2], r8w
    shl r8d, 16
    lea rax, [r32]
    mov [rax+rbx*4], r8d
    movd xmm0, r8d
    cvtss2sd xmm0, xmm0
    lea rax, [r64]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, N2
    jb .gr

    say "the cpu version, f64", 13, 10
    call cpu_fwd
    call cpu_bwd
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    lea rcx, [w164]
    lea rdx, [dw164]
    lea r8, [rsp+32]
    call fdw
    lea rcx, [w264]
    lea rdx, [dw264]
    lea r8, [rsp+32]
    call fdw
    call cpu_fwd
    movsd xmm0, [rsp+32]
    close_to 0.0, 1e-6, "  formula vs finite differences, 20 weights of each conv, max diff"

    say "the gpu kernels in f32 vs the cpu", 13, 10
    mov ecx, NX * 4
    call galloc
    mov [d_x], rax
    mov ecx, NW * 4
    call galloc
    mov [d_w1], rax
    mov ecx, NW * 4
    call galloc
    mov [d_w2], rax
    mov ecx, N2 * 4
    call galloc
    mov [d_r], rax
    mov ecx, N2 * 2
    call galloc
    mov [d_rb], rax
    mov ecx, N1 * 4
    call galloc
    mov [d_a1], rax
    mov ecx, N2 * 4
    call galloc
    mov [d_b2], rax
    mov ecx, N2 * 4
    call galloc
    mov [d_a2], rax
    mov ecx, N2 * 4
    call galloc
    mov [d_db2], rax
    mov ecx, N1 * 4
    call galloc
    mov [d_da1], rax
    mov ecx, N1 * 4
    call galloc
    mov [d_dp1], rax
    mov ecx, NCH * NW * 4
    call galloc
    mov [d_part], rax
    mov ecx, NW * 4
    call galloc
    mov [d_dw1], rax
    mov ecx, NW * 4
    call galloc
    mov [d_dw2], rax
    mov rcx, [d_x]
    lea rdx, [x32]
    mov r8d, NX * 4
    call gpu_up
    mov rcx, [d_w1]
    lea rdx, [w132]
    mov r8d, NW * 4
    call gpu_up
    mov rcx, [d_w2]
    lea rdx, [w232]
    mov r8d, NW * 4
    call gpu_up
    mov rcx, [d_r]
    lea rdx, [r32]
    mov r8d, N2 * 4
    call gpu_up
    mov rcx, [d_rb]
    lea rdx, [rb]
    mov r8d, N2 * 2
    call gpu_up
    mov dword [s_b], NB
    mov dword [s_c], C
    mov dword [s_bf], 0
    mov dword [s_ch], NCH
    call gpu_chain
    mov rcx, [d_a1]
    lea rdx, [a164]
    mov r8d, N1
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv1 (full, from the mel) + relu, max diff over the largest"
    mov rcx, [d_b2]
    lea rdx, [b264]
    mov r8d, N2
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv2 (depthwise)"
    mov rcx, [d_a2]
    lea rdx, [a264]
    mov r8d, N2
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  relu"
    mov rcx, [d_db2]
    lea rdx, [db264]
    mov r8d, N2
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  relu backward"
    mov rcx, [d_da1]
    lea rdx, [da164]
    mov r8d, N1
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv2 backward, inputs"
    mov rcx, [d_dw2]
    lea rdx, [dw264]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv2 backward, taps and bias"
    mov rcx, [d_dp1]
    lea rdx, [dp164]
    mov r8d, N1
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  relu backward through conv1's output"
    mov rcx, [d_dw1]
    lea rdx, [dw164]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 1e-5, "  conv1 backward, taps and bias"

    say "the same in bf16 (rounding on every T tensor)", 13, 10
    mov dword [s_bf], 1
    call gpu_chain
    mov rcx, [d_a1]
    lea rdx, [a164]
    mov r8d, N1
    mov r9d, 1
    call against
    close_to 0.0, 1e-2, "  conv1 + relu"
    mov rcx, [d_da1]
    lea rdx, [da164]
    mov r8d, N1
    mov r9d, 1
    call against
    close_to 0.0, 2e-2, "  conv2 backward, inputs"
    mov rcx, [d_dw2]
    lea rdx, [dw264]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 2e-2, "  conv2 backward, taps and bias"
    mov rcx, [d_dw1]
    lea rdx, [dw164]
    mov r8d, NW
    xor r9d, r9d
    call against
    close_to 0.0, 2e-2, "  conv1 backward, taps and bias"

    say "speed", 13, 10
    call big
    jmp t_done

; the real size, bf16: 32 x 15 s of 80-bin mel through the three convs (the
; pointwise matmuls between them aren't in here). 10 times each way
big:
    push rbx
    sub rsp, 48
    mov ecx, BB * BT0 * BF0 * 4
    call galloc
    mov [d_x], rax
    mov ecx, 10 * BC * 4
    call galloc
    mov [d_w1], rax
    mov ecx, 10 * BC * 4
    call galloc
    mov [d_w2], rax
    mov ecx, 10 * BC * 4
    call galloc
    mov [d_dw1], rax
    mov ecx, BCH * 10 * BC * 4
    call galloc
    mov [d_part], rax
    mov ecx, BB * BT1 * BF1 * BC * 2
    call galloc
    mov [d_a1], rax
    mov ecx, BB * BT1 * BF1 * BC * 2
    call galloc
    mov [d_da1], rax
    mov ecx, BB * BT2 * BF2 * BC * 2
    call galloc
    mov [d_a2], rax
    mov ecx, BB * BT2 * BF2 * BC * 2
    call galloc
    mov [d_da2], rax
    mov ecx, BB * BT3 * BF3 * BC * 2
    call galloc
    mov [d_a3], rax
    mov ecx, BB * BT3 * BF3 * BC * 2
    call galloc
    mov [d_da3], rax
    mov dword [s_b], BB
    mov dword [s_c], BC
    mov dword [s_bf], 1
    mov dword [s_ch], BCH
    call big_fwd                ; warm up
    call big_bwd
    xor ecx, ecx
    call gpu_tstart
    mov ebx, 10
.f:
    call big_fwd
    dec ebx
    jnz .f
    xor ecx, ecx
    call gpu_tstop
    mov eax, 10
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+32], xmm0
    xor ecx, ecx
    call gpu_tstart
    mov ebx, 10
.b:
    call big_bwd
    dec ebx
    jnz .b
    xor ecx, ecx
    call gpu_tstop
    mov eax, 10
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+40], xmm0
    say "  32 x 15 s, 80 bins, 256 channels, bf16: forward "
    movsd xmm0, [rsp+32]
    mov edx, 2
    call print_fixed
    say " ms, backward "
    movsd xmm0, [rsp+40]
    mov edx, 2
    call print_fixed
    say " ms (convs only)", 13, 10
    add rsp, 48
    pop rbx
    ret

big_fwd:
    sub rsp, 40
    mov dword [s_t], BT0
    mov dword [s_f], BF0
    mov dword [s_fl], 6
    mov rcx, [d_x]
    mov rdx, [d_w1]
    mov r8, [d_a1]
    call run_sconv
    mov dword [s_t], BT1
    mov dword [s_f], BF1
    mov dword [s_fl], 0
    mov rcx, [d_a1]
    mov rdx, [d_w2]
    mov r8, [d_a2]
    call run_sconv
    mov dword [s_t], BT2
    mov dword [s_f], BF2
    mov rcx, [d_a2]
    mov rdx, [d_w2]
    mov r8, [d_a3]
    call run_sconv
    add rsp, 40
    ret

big_bwd:
    sub rsp, 40
    mov dword [s_t], BT2
    mov dword [s_f], BF2
    mov dword [s_fl], 0
    mov rcx, [d_da3]
    mov rdx, [d_w2]
    mov r8, [d_da2]
    call run_sbx
    mov rcx, [d_da3]
    mov rdx, [d_a2]
    mov r8, [d_part]
    mov r9, [d_dw1]
    call run_sdw
    mov dword [s_t], BT1
    mov dword [s_f], BF1
    mov rcx, [d_da2]
    mov rdx, [d_w2]
    mov r8, [d_da1]
    call run_sbx
    mov rcx, [d_da2]
    mov rdx, [d_a1]
    mov r8, [d_part]
    mov r9, [d_dw1]
    call run_sdw
    mov rcx, [d_a1]
    mov rdx, [d_da1]
    mov r8, [d_da1]
    mov r9d, BB * BT1 * BF1 * BC
    call run_relub
    mov dword [s_t], BT0
    mov dword [s_f], BF0
    mov dword [s_fl], 2
    mov rcx, [d_da1]
    mov rdx, [d_x]
    mov r8, [d_part]
    mov r9, [d_dw1]
    call run_sdw
    add rsp, 40
    ret

section .rdata
align 8
c_one   dq 1.0
