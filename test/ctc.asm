; stage 4 test: the ctc loss. a plain f64 version on the cpu, its gradient checked
; by finite differences, then the gpu kernels (lsm, ctc_alpha, ctc_beta, ctc_grad)
; checked against it: losses, gradients, the impossible utterance, padding rows
; uses: gpu\cuda gpu\kernels
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "gpu/cuda.inc"

extern ptx_ctc, ptx_ctc_end

NB      equ 6
TMAX    equ 50
V       equ 300
UMAX    equ 20
SMAX    equ 2 * UMAX + 1
BLANK   equ V - 1
; the big one, for speed
BB      equ 64
BIGT    equ 400
BV      equ 1024
BU      equ 150

section .rdata
k_lsm   db "lsm", 0
k_alpha db "ctc_alpha", 0
k_beta  db "ctc_beta", 0
k_grad  db "ctc_grad", 0
; the utterances: frames, labels
tl      dd 50, 12, 1, 30, 4, 3
ul      dd 20, 5, 0, 1, 4, 3
lab1    dd 3, 3, 3, 7, 7              ; repeats need blanks between them
lab4    dd 5, 6, 7, 8                 ; exactly as many frames as labels
lab5    dd 1, 1, 1                    ; needs 5 frames, gets 3: impossible
align 8
c_ninf  dq 0xfff0000000000000
c_three dq 3.0
c_eps   dq 1e-4
c_two   dq 2.0
c_half  dq 0.5
c_one   dq 1.0
c_ms    dq 1000.0
align 16
c_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff

section .bss
alignb 8
kl      resb KL_SIZE
f_lsm   resq 1
f_alpha resq 1
f_beta  resq 1
f_grad  resq 1
rng     resb RNG_SIZE
logits  resd NB * TMAX * V          ; f32, what the gpu gets
labels  resd NB * UMAX
x64     resq NB * TMAX * V          ; f64 copies for the cpu
lp      resq NB * TMAX * V
alp     resq TMAX * SMAX
bet     resq TMAX * SMAX
closs   resq NB
cgrad   resq NB * TMAX * V
gloss   resd NB
ggrad   resd NB * TMAX * V
scale   resq 1
d_x     resq 1
d_lp    resq 1
d_lab   resq 1
d_tl    resq 1
d_ul    resq 1
d_al    resq 1
d_be    resq 1
d_loss  resq 1
d_g     resq 1

section .text

; ---- the cpu version, f64, straight from the definition

; xmm0, xmm1 = a, b. xmm0 = ln(e^a + e^b), -inf safe (math_exp isn't: it does
; -inf - -inf inside, so a -inf term never goes near it)
lse2:
    sub rsp, 40
    ucomisd xmm0, [c_ninf]
    jne .anum
    movapd xmm0, xmm1           ; ln(0 + e^b) = b
    add rsp, 40
    ret
.anum:
    ucomisd xmm1, [c_ninf]
    jne .bnum
    add rsp, 40                 ; ln(e^a + 0) = a
    ret
.bnum:
    movapd xmm2, xmm0
    maxsd xmm2, xmm1
    ucomisd xmm2, [c_ninf]
    je .ninf
    movsd [rsp+32], xmm2
    subsd xmm0, xmm2
    subsd xmm1, xmm2
    movsd [rsp+24], xmm1
    call math_exp
    movsd [rsp+16], xmm0
    movsd xmm0, [rsp+24]
    call math_exp
    addsd xmm0, [rsp+16]
    call math_log
    addsd xmm0, [rsp+32]
    add rsp, 40
    ret
.ninf:
    movsd xmm0, [c_ninf]
    add rsp, 40
    ret

; ecx = utterance, edx = state. eax = the label that state emits
slab:
    test edx, 1
    jz .blank
    imul ecx, ecx, UMAX
    shr edx, 1
    add ecx, edx
    lea rax, [labels]
    mov eax, [rax+rcx*4]
    ret
.blank:
    mov eax, BLANK
    ret

; x64 -> lp, log softmax per row (rows of every utterance, padding included)
cpu_lsm:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    xor ebx, ebx                ; row
.row:
    cmp ebx, NB * TMAX
    jae .out
    imul esi, ebx, V
    ; max, then ln sum exp
    lea rax, [x64]
    lea rax, [rax+rsi*8]        ; the row
    movsd xmm0, [rax]
    mov ecx, 1
.mx:
    maxsd xmm0, [rax+rcx*8]
    inc ecx
    cmp ecx, V
    jb .mx
    movsd [rsp+32], xmm0
    xorpd xmm1, xmm1
    movsd [rsp+40], xmm1
    xor edi, edi
.se:
    lea rax, [x64]
    lea rcx, [rsi+rdi]
    movsd xmm0, [rax+rcx*8]
    subsd xmm0, [rsp+32]
    call math_exp
    addsd xmm0, [rsp+40]
    movsd [rsp+40], xmm0
    inc edi
    cmp edi, V
    jb .se
    movsd xmm0, [rsp+40]
    call math_log
    addsd xmm0, [rsp+32]        ; the row's lse
    xor edi, edi
.w:
    lea rax, [x64]
    lea rcx, [rsi+rdi]
    movsd xmm1, [rax+rcx*8]
    subsd xmm1, xmm0
    lea rax, [lp]
    movsd [rax+rcx*8], xmm1
    inc edi
    cmp edi, V
    jb .w
    inc ebx
    jmp .row
.out:
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; logp of utterance ecx, frame edx, label r8d -> xmm0
%macro getlp 0
    imul eax, ecx, TMAX
    add eax, edx
    imul eax, eax, V
    add eax, r8d
    lea r9, [lp]
    movsd xmm0, [r9+rax*8]
%endmacro

; ecx = utterance. fills al and be, and closs[b] = -ln p. the plain recursions
cpu_ctc:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    mov r12d, ecx               ; b
    lea rax, [tl]
    mov r13d, [rax+r12*4]       ; T
    lea rax, [ul]
    mov r14d, [rax+r12*4]
    lea r14d, [r14*2+1]         ; S
    ; alpha
    xor esi, esi                ; t
.at:
    cmp esi, r13d
    jae .adone
    xor edi, edi                ; s
.as:
    cmp edi, r14d
    jae .anext
    mov ecx, r12d
    mov edx, edi
    call slab
    mov r15d, eax               ; its label
    test esi, esi
    jnz .arec
    movsd xmm0, [c_ninf]
    cmp edi, 1
    ja .aput
    mov ecx, r12d
    xor edx, edx
    mov r8d, r15d
    getlp
    jmp .aput
.arec:
    ; lse(alpha[t-1][s], alpha[t-1][s-1], alpha[t-1][s-2] if allowed) + lp
    lea eax, [rsi-1]
    imul eax, eax, SMAX
    add eax, edi
    lea r9, [alp]
    lea rbx, [r9+rax*8]         ; alpha[t-1][s]
    movsd xmm0, [rbx]
    movsd xmm1, [c_ninf]
    test edi, edi
    jz .a1
    movsd xmm1, [rbx-8]
.a1:
    call lse2
    movsd [rsp+32], xmm0
    cmp edi, 2
    jb .a2
    cmp r15d, BLANK
    je .a2
    mov ecx, r12d
    lea edx, [rdi-2]
    call slab
    cmp eax, r15d
    je .a2
    movsd xmm0, [rsp+32]
    movsd xmm1, [rbx-16]
    call lse2
    movsd [rsp+32], xmm0
.a2:
    mov ecx, r12d
    mov edx, esi
    mov r8d, r15d
    getlp
    addsd xmm0, [rsp+32]
.aput:
    imul eax, esi, SMAX
    add eax, edi
    lea r9, [alp]
    movsd [r9+rax*8], xmm0
    inc edi
    jmp .as
.anext:
    inc esi
    jmp .at
.adone:
    ; -ln(alpha[T-1][S-1] + alpha[T-1][S-2])
    lea eax, [r13-1]
    imul eax, eax, SMAX
    add eax, r14d
    lea r9, [alp]
    movsd xmm0, [r9+rax*8-8]
    movsd xmm1, [c_ninf]
    cmp r14d, 2
    jb .l1
    movsd xmm1, [r9+rax*8-16]
.l1:
    call lse2
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    lea rax, [closs]
    movsd [rax+r12*8], xmm1
    ; beta, from the end
    lea esi, [r13-1]
.bt:
    xor edi, edi
.bs:
    cmp edi, r14d
    jae .bnext
    mov ecx, r12d
    mov edx, edi
    call slab
    mov r15d, eax
    lea eax, [r13-1]
    cmp esi, eax
    jne .brec
    movsd xmm0, [c_ninf]
    lea eax, [r14-2]
    cmp edi, eax
    jl .bput
    mov ecx, r12d
    mov edx, esi
    mov r8d, r15d
    getlp
    jmp .bput
.brec:
    lea eax, [rsi+1]
    imul eax, eax, SMAX
    add eax, edi
    lea r9, [bet]
    lea rbx, [r9+rax*8]         ; beta[t+1][s]
    movsd xmm0, [rbx]
    movsd xmm1, [c_ninf]
    lea eax, [rdi+1]
    cmp eax, r14d
    jae .b1
    movsd xmm1, [rbx+8]
.b1:
    call lse2
    movsd [rsp+32], xmm0
    lea eax, [rdi+2]
    cmp eax, r14d
    jae .b2
    cmp r15d, BLANK
    je .b2
    mov ecx, r12d
    lea edx, [rdi+2]
    call slab
    cmp eax, r15d
    je .b2
    movsd xmm0, [rsp+32]
    movsd xmm1, [rbx+16]
    call lse2
    movsd [rsp+32], xmm0
.b2:
    mov ecx, r12d
    mov edx, esi
    mov r8d, r15d
    getlp
    addsd xmm0, [rsp+32]
.bput:
    imul eax, esi, SMAX
    add eax, edi
    lea r9, [bet]
    movsd [r9+rax*8], xmm0
    inc edi
    jmp .bs
.bnext:
    dec esi
    jns .bt
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = utterance (after cpu_ctc). cgrad rows = scale (softmax - posterior shares)
cpu_grad:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov r12d, ecx
    lea rax, [tl]
    mov r13d, [rax+r12*4]
    lea rax, [ul]
    mov r14d, [rax+r12*4]
    lea r14d, [r14*2+1]
    lea rax, [closs]
    movsd xmm0, [rax+r12*8]
    movsd [rsp+32], xmm0        ; -ln p
    xor esi, esi                ; t, over the whole padded length
.t:
    cmp esi, TMAX
    jae .out
    imul ebx, r12d, TMAX
    add ebx, esi
    imul ebx, ebx, V            ; the row's start
    ; padding or impossible: zeros
    cmp esi, r13d
    jae .zero
    movsd xmm0, [rsp+32]
    andpd xmm0, [c_abs]
    ucomisd xmm0, [c_ninf+0]    ; (|loss| vs -inf: just a not-nan check)
    jp .zero
    mov rax, 0x7ff0000000000000
    movq xmm1, rax
    ucomisd xmm0, xmm1
    jae .zero
    xor edi, edi
.sm:
    lea rax, [lp]
    lea rcx, [rbx+rdi]
    movsd xmm0, [rax+rcx*8]
    call math_exp
    mulsd xmm0, [scale]
    lea rax, [cgrad]
    lea rcx, [rbx+rdi]
    movsd [rax+rcx*8], xmm0
    inc edi
    cmp edi, V
    jb .sm
    xor edi, edi                ; state
.st:
    cmp edi, r14d
    jae .nextt
    mov ecx, r12d
    mov edx, edi
    call slab
    mov r15d, eax
    imul eax, esi, SMAX
    add eax, edi
    lea r9, [alp]
    movsd xmm0, [r9+rax*8]
    lea r9, [bet]
    addsd xmm0, [r9+rax*8]
    ucomisd xmm0, [c_ninf]      ; an unreachable state has no share (and math_exp
    je .nextst                  ; can't take -inf)
    lea rax, [lp]
    lea rcx, [rbx+r15]
    subsd xmm0, [rax+rcx*8]
    addsd xmm0, [rsp+32]
    call math_exp
    mulsd xmm0, [scale]
    lea rax, [cgrad]
    lea rcx, [rbx+r15]
    movsd xmm1, [rax+rcx*8]
    subsd xmm1, xmm0
    movsd [rax+rcx*8], xmm1
.nextst:
    inc edi
    jmp .st
.zero:
    xor edi, edi
    lea rax, [cgrad]
.z:
    lea rcx, [rbx+rdi]
    mov qword [rax+rcx*8], 0
    inc edi
    cmp edi, V
    jb .z
.nextt:
    inc esi
    jmp .t
.out:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the cpu loss of utterance ecx from scratch (for finite differences). xmm0
cpu_loss:
    push rbx
    sub rsp, 32
    mov ebx, ecx
    call cpu_lsm
    mov ecx, ebx
    call cpu_ctc
    lea rax, [closs]
    movsd xmm0, [rax+rbx*8]
    add rsp, 32
    pop rbx
    ret

; ---- the gpu side

; runs the four kernels on whatever is in d_x etc. with NB/TMAX/V/UMAX
gpu_run:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov rax, [f_lsm]
    mov [rbx+KL_FUNC], rax
    grid NB * TMAX, 1, 256, 1
    mov rax, [d_x]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lp]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], V
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_alpha]
    mov [rbx+KL_FUNC], rax
    grid NB, 1, 1024, 1
    mov rax, [d_lp]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lab]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_tl]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_ul]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_al]
    mov [rbx+KL_ARGS+32], rax
    mov rax, [d_loss]
    mov [rbx+KL_ARGS+40], rax
    mov qword [rbx+KL_ARGS+48], TMAX
    mov qword [rbx+KL_ARGS+56], V
    mov qword [rbx+KL_ARGS+64], UMAX
    mov qword [rbx+KL_ARGS+72], BLANK
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_beta]
    mov [rbx+KL_FUNC], rax
    mov rax, [d_be]
    mov [rbx+KL_ARGS+32], rax
    mov qword [rbx+KL_ARGS+40], TMAX
    mov qword [rbx+KL_ARGS+48], V
    mov qword [rbx+KL_ARGS+56], UMAX
    mov qword [rbx+KL_ARGS+64], BLANK
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_grad]
    mov [rbx+KL_FUNC], rax
    grid NB * TMAX, 1, 256, 1
    mov rax, [d_lp]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lab]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_tl]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_ul]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_al]
    mov [rbx+KL_ARGS+32], rax
    mov rax, [d_be]
    mov [rbx+KL_ARGS+40], rax
    mov rax, [d_loss]
    mov [rbx+KL_ARGS+48], rax
    mov rax, [d_g]
    mov [rbx+KL_ARGS+56], rax
    mov qword [rbx+KL_ARGS+64], TMAX
    mov qword [rbx+KL_ARGS+72], V
    mov qword [rbx+KL_ARGS+80], UMAX
    mov qword [rbx+KL_ARGS+88], BLANK
    cvtsd2ss xmm0, [scale]
    movd eax, xmm0
    mov [rbx+KL_ARGS+96], rax
    mov rcx, rbx
    call gpu_launch
    call gpu_sync
    add rsp, 32
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
    lea rcx, [ptx_ctc]
    lea rdx, [ptx_ctc_end]
    sub rdx, rcx
    call gpu_module
    mov rbx, rax
    mov rcx, rbx
    lea rdx, [k_lsm]
    call gpu_func
    mov [f_lsm], rax
    mov rcx, rbx
    lea rdx, [k_alpha]
    call gpu_func
    mov [f_alpha], rax
    mov rcx, rbx
    lea rdx, [k_beta]
    call gpu_func
    mov [f_beta], rax
    mov rcx, rbx
    lea rdx, [k_grad]
    call gpu_func
    mov [f_grad], rax

    ; random logits (normal, x3) and labels from a small alphabet, so repeats happen
    lea rcx, [rng]
    mov edx, 17
    call rng_seed
    xor ebx, ebx
.lg:
    lea rcx, [rng]
    call rng_normal
    mulsd xmm0, [c_three]
    cvtsd2ss xmm1, xmm0
    lea rax, [logits]
    movss [rax+rbx*4], xmm1
    cvtss2sd xmm1, xmm1         ; the cpu sees exactly what the gpu sees
    lea rax, [x64]
    movsd [rax+rbx*8], xmm1
    inc ebx
    cmp ebx, NB * TMAX * V
    jb .lg
    xor ebx, ebx
.lb:
    lea rcx, [rng]
    call rng_next
    and eax, 7
    lea rcx, [labels]
    mov [rcx+rbx*4], eax
    inc ebx
    cmp ebx, NB * UMAX
    jb .lb
    ; the hand-made ones
    lea rdi, [labels+1*UMAX*4]
    lea rsi, [lab1]
    mov ecx, 5
    rep movsd
    lea rdi, [labels+4*UMAX*4]
    lea rsi, [lab4]
    mov ecx, 4
    rep movsd
    lea rdi, [labels+5*UMAX*4]
    lea rsi, [lab5]
    mov ecx, 3
    rep movsd
    ; scale = 1 / labels of the possible utterances
    xor eax, eax
    lea rcx, [ul]
    add eax, [rcx]
    add eax, [rcx+4]
    add eax, [rcx+8]
    add eax, [rcx+12]
    add eax, [rcx+16]
    cvtsi2sd xmm1, eax
    movsd xmm0, [c_one]
    divsd xmm0, xmm1
    movsd [scale], xmm0

    say "the cpu version, f64", 13, 10
    call cpu_lsm
    xor r12d, r12d
.cu:
    mov ecx, r12d
    call cpu_ctc
    mov ecx, r12d
    call cpu_grad
    inc r12d
    cmp r12d, NB
    jb .cu
    lea rax, [closs]
    movsd xmm0, [rax+5*8]
    mov rax, 0x7ff0000000000000
    movq xmm1, rax
    ucomisd xmm0, xmm1
    sete cl
    movzx ecx, cl
    lea rdx, [s_inf]
    movapd xmm2, xmm0
    call t_okf
    ; finite differences: a few logits of utterances 0, 1, 3, 4, d(-ln p)/dx vs the
    ; formula (scaled back up)
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    xor r13d, r13d
.fd:
    cmp r13d, 40
    jae .fddone
    lea rcx, [rng]
    call rng_next
    mov rsi, rax
    ; utterance: one of 0, 1, 3, 4
    mov eax, esi
    and eax, 3
    lea rcx, [pick]
    movzx r12d, byte [rcx+rax]
    ; frame under its T, any vocab entry
    mov rax, rsi
    shr rax, 8
    lea rcx, [tl]
    mov ecx, [rcx+r12*4]
    xor edx, edx
    div rcx
    mov edi, edx                ; t
    mov rax, rsi
    shr rax, 32
    xor edx, edx
    mov ecx, V
    div rcx
    imul eax, r12d, TMAX
    add eax, edi
    imul eax, eax, V
    add eax, edx
    mov ebx, eax                ; the logit
    lea rax, [x64]
    movsd xmm0, [rax+rbx*8]
    movsd [rsp+40], xmm0
    addsd xmm0, [c_eps]
    movsd [rax+rbx*8], xmm0
    mov ecx, r12d
    call cpu_loss
    movsd [rsp+24], xmm0
    lea rax, [x64]
    movsd xmm0, [rsp+40]
    subsd xmm0, [c_eps]
    movsd [rax+rbx*8], xmm0
    mov ecx, r12d
    call cpu_loss
    movsd xmm1, [rsp+24]
    subsd xmm1, xmm0
    divsd xmm1, [c_eps]
    mulsd xmm1, [c_half]        ; central difference
    lea rax, [x64]
    movsd xmm0, [rsp+40]
    movsd [rax+rbx*8], xmm0
    lea rax, [cgrad]
    movsd xmm0, [rax+rbx*8]
    divsd xmm0, [scale]
    subsd xmm0, xmm1
    andpd xmm0, [c_abs]
    movsd xmm1, [rsp+32]
    maxsd xmm1, xmm0            ; new value second: a nan in it carries through
    movsd [rsp+32], xmm1
    inc r13d
    jmp .fd
.fddone:
    call cpu_lsm                ; put lp back as it was
    movsd xmm0, [rsp+32]
    close_to 0.0, 1e-7, "  formula vs finite differences, 40 logits, max diff"

    say "the gpu kernels vs the cpu", 13, 10
    mov ecx, NB * TMAX * V * 4
    call gpu_alloc
    mov [d_x], rax
    mov ecx, NB * TMAX * V * 4
    call gpu_alloc
    mov [d_lp], rax
    mov ecx, NB * TMAX * V * 4
    call gpu_alloc
    mov [d_g], rax
    mov ecx, NB * UMAX * 4
    call gpu_alloc
    mov [d_lab], rax
    mov ecx, NB * 4
    call gpu_alloc
    mov [d_tl], rax
    mov ecx, NB * 4
    call gpu_alloc
    mov [d_ul], rax
    mov ecx, NB * 4
    call gpu_alloc
    mov [d_loss], rax
    mov ecx, NB * TMAX * SMAX * 4
    call gpu_alloc
    mov [d_al], rax
    mov ecx, NB * TMAX * SMAX * 4
    call gpu_alloc
    mov [d_be], rax
    mov rcx, [d_x]
    lea rdx, [logits]
    mov r8d, NB * TMAX * V * 4
    call gpu_up
    mov rcx, [d_lab]
    lea rdx, [labels]
    mov r8d, NB * UMAX * 4
    call gpu_up
    mov rcx, [d_tl]
    lea rdx, [tl]
    mov r8d, NB * 4
    call gpu_up
    mov rcx, [d_ul]
    lea rdx, [ul]
    mov r8d, NB * 4
    call gpu_up
    call gpu_run
    lea rcx, [gloss]
    mov rdx, [d_loss]
    mov r8d, NB * 4
    call gpu_down
    lea rcx, [ggrad]
    mov rdx, [d_g]
    mov r8d, NB * TMAX * V * 4
    call gpu_down
    ; losses, relative
    xorpd xmm2, xmm2
    xor ecx, ecx
.ls:
    lea rax, [gloss]
    cvtss2sd xmm0, [rax+rcx*4]
    lea rax, [closs]
    movsd xmm1, [rax+rcx*8]
    subsd xmm0, xmm1
    divsd xmm0, xmm1
    andpd xmm0, [c_abs]
    maxsd xmm2, xmm0
    inc ecx
    cmp ecx, 5                  ; the 6th is the impossible one
    jb .ls
    movapd xmm0, xmm2
    close_to 0.0, 1e-5, "  losses of the 5 possible utterances, max relative diff"
    lea rax, [gloss]
    mov eax, [rax+5*4]
    cmp eax, 0x7f800000
    check e, "  the impossible one: +inf"
    ; gradients, every entry, relative to the largest
    xorpd xmm2, xmm2
    xorpd xmm3, xmm3
    xor ecx, ecx
.gs:
    lea rax, [ggrad]
    cvtss2sd xmm0, [rax+rcx*4]
    lea rax, [cgrad]
    movsd xmm1, [rax+rcx*8]
    movapd xmm4, xmm1
    andpd xmm4, [c_abs]
    maxsd xmm3, xmm4
    subsd xmm0, xmm1
    andpd xmm0, [c_abs]
    maxsd xmm2, xmm0
    inc ecx
    cmp ecx, NB * TMAX * V
    jb .gs
    divsd xmm2, xmm3
    movapd xmm0, xmm2
    close_to 0.0, 5e-4, "  every gradient entry, max diff over the largest (f32 log space)"
    ; padding rows and the impossible utterance are exactly 0
    xor eax, eax
    xor ecx, ecx
.zs:
    lea rdx, [ggrad]
    or eax, [rdx+(2*TMAX+1)*V*4+rcx*4]  ; utterance 2 has 1 frame: frame 1 on is padding
    or eax, [rdx+(5*TMAX)*V*4+rcx*4]    ; the impossible one's first frame
    inc ecx
    cmp ecx, V
    jb .zs
    cmp eax, 0
    check e, "  padding and the impossible utterance get zero gradient"

    say "speed", 13, 10
    call big
    jmp t_done

; the big case: 64 utterances of 400 frames, 150 labels, a 1024 vocab
big:
    push rbx
    push rsi
    sub rsp, 56
    mov rcx, BB * BIGT * BV * 4
    call gpu_alloc
    mov [d_x], rax
    mov rcx, BB * BIGT * BV * 4
    call gpu_alloc
    mov [d_lp], rax
    mov rcx, BB * BIGT * BV * 4
    call gpu_alloc
    mov [d_g], rax
    mov ecx, BB * BU * 4
    call gpu_alloc
    mov [d_lab], rax
    mov ecx, BB * BIGT * (2 * BU + 1) * 4
    call gpu_alloc
    mov [d_al], rax
    mov ecx, BB * BIGT * (2 * BU + 1) * 4
    call gpu_alloc
    mov [d_be], rax
    mov ecx, BB * 4
    call gpu_alloc
    mov [d_tl], rax
    mov ecx, BB * 4
    call gpu_alloc
    mov [d_ul], rax
    mov ecx, BB * 4
    call gpu_alloc
    mov [d_loss], rax
    ; lengths and labels: any values, the timing doesn't care
    mov ecx, BB * BU * 4
    call mem_alloc
    mov rsi, rax
    xor ecx, ecx
.lab:
    mov eax, ecx
    and eax, 511
    mov [rsi+rcx*4], eax
    inc ecx
    cmp ecx, BB * BU
    jb .lab
    mov rcx, [d_lab]
    mov rdx, rsi
    mov r8d, BB * BU * 4
    call gpu_up
    xor ecx, ecx
.len:
    mov dword [rsi+rcx*4], BIGT
    mov dword [rsi+rcx*4+BB*4], BU
    inc ecx
    cmp ecx, BB
    jb .len
    mov rcx, [d_tl]
    mov rdx, rsi
    mov r8d, BB * 4
    call gpu_up
    mov rcx, [d_ul]
    lea rdx, [rsi+BB*4]
    mov r8d, BB * 4
    call gpu_up
    ; zero logits: a uniform softmax is as good as any for timing
    mov rcx, BB * BIGT * BV * 4
    call mem_alloc
    mov rcx, [d_x]
    mov rdx, rax
    mov r8, BB * BIGT * BV * 4
    call gpu_up
    lea rbx, [kl]
    call time_now
    mov [rsp+32], rax
    mov esi, 10
.rep:
    mov rax, [f_lsm]
    mov [rbx+KL_FUNC], rax
    grid BB * BIGT, 1, 256, 1
    mov rax, [d_x]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lp]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], BV
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_alpha]
    mov [rbx+KL_FUNC], rax
    grid BB, 1, 1024, 1
    mov rax, [d_lp]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lab]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_tl]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_ul]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_al]
    mov [rbx+KL_ARGS+32], rax
    mov rax, [d_loss]
    mov [rbx+KL_ARGS+40], rax
    mov qword [rbx+KL_ARGS+48], BIGT
    mov qword [rbx+KL_ARGS+56], BV
    mov qword [rbx+KL_ARGS+64], BU
    mov qword [rbx+KL_ARGS+72], BV - 1
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_beta]
    mov [rbx+KL_FUNC], rax
    mov rax, [d_be]
    mov [rbx+KL_ARGS+32], rax
    mov qword [rbx+KL_ARGS+40], BIGT
    mov qword [rbx+KL_ARGS+48], BV
    mov qword [rbx+KL_ARGS+56], BU
    mov qword [rbx+KL_ARGS+64], BV - 1
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_grad]
    mov [rbx+KL_FUNC], rax
    grid BB * BIGT, 1, 256, 1
    mov rax, [d_lp]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lab]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_tl]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_ul]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_al]
    mov [rbx+KL_ARGS+32], rax
    mov rax, [d_be]
    mov [rbx+KL_ARGS+40], rax
    mov rax, [d_loss]
    mov [rbx+KL_ARGS+48], rax
    mov rax, [d_g]
    mov [rbx+KL_ARGS+56], rax
    mov qword [rbx+KL_ARGS+64], BIGT
    mov qword [rbx+KL_ARGS+72], BV
    mov qword [rbx+KL_ARGS+80], BU
    mov qword [rbx+KL_ARGS+88], BV - 1
    mov qword [rbx+KL_ARGS+96], 0x3f800000
    mov rcx, rbx
    call gpu_launch
    dec esi
    jnz .rep
    call gpu_sync
    mov rcx, [rsp+32]
    call time_since
    mulsd xmm0, [c_ms]
    mov eax, 10
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+40], xmm0
    say "  64 x 400 frames, 150 labels, 1024 vocab: lsm + alpha + beta + grad in "
    movsd xmm0, [rsp+40]
    mov edx, 2
    call print_fixed
    say " ms", 13, 10
    add rsp, 56
    pop rsi
    pop rbx
    ret

section .rdata
s_inf   db "  the impossible utterance's loss is +inf", 0
pick    db 0, 1, 3, 4
