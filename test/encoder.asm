; stage 4 test: the whole encoder in f32 on the naive kernels. a tiny model with
; awkward sizes (front end, two blocks, the ctc head), random mel and labels, a
; chunked attention mask.
;  - padding doesn't leak: wild values in an utterance's padded mel frames leave
;    every loss exactly as it was
;  - for every parameter tensor, nudge it along a direction u and compare
;    (L(w + eu) - L(w - eu)) / 2e with g.u, along the gradient itself and along a
;    random direction (the method of mnemonic's test\grad)
;  - a few adamw steps on the one batch: the loss has to fall
; uses: gpu\cuda gpu\kernels asr\model
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "asr/model.inc"
%include "test/check.inc"

TL      equ 2                   ; blocks
TD      equ 48
TH      equ 4                   ; heads of 12
TF      equ 40
TV      equ 37
TK      equ 5                   ; depthwise taps
TC0     equ 8                   ; subsampling channels
TMEL    equ 20                  ; bins: 10, 5, 3 after the convs
TT      equ 37                  ; mel frames: 19, 10, 5 after the convs
TB      equ 3
TU      equ 3                   ; labels at most
TF3     equ 3
NMELS   equ TB * TT * TMEL

section .rdata
align 4
tls     dd 5, 4, 2              ; encoder frames per utterance
uls     dd 3, 2, 1              ; labels: 6 in all
align 8
c_eps   dq 1e-3
c_shift dq 2e-4                 ; aim for a loss change this big, well above f32 noise
c_tol   dq 1e-2
c_half  dq 0.5
c_scale dq 0.16666666666666666  ; 1/6 labels
c_wild  dq 1000.0
; adam for the overfit: lr, b1, b2, eps, wd, clip
op      dq 3e-3, 0.9, 0.99, 1e-8, 0.0, 1e9

n_wl    db "front end wl", 0
n_wp2   db "front end wp2", 0
n_wp3   db "front end wp3", 0
n_13a   db "ff1 w1/w3", 0
n_2a    db "ff1 w2", 0
n_qkv   db "wqkv", 0
n_o     db "wo", 0
n_pw1   db "conv pointwise in", 0
n_pw2   db "conv pointwise out", 0
n_13b   db "ff2 w1/w3", 0
n_2b    db "ff2 w2", 0
n_head  db "ctc head", 0
n_w1    db "front end conv1 taps", 0
n_w2    db "front end conv2 taps", 0
n_w3    db "front end conv3 taps", 0
n_nfa   db "ff1 norm", 0
n_nat   db "attn norm", 0
n_ncv   db "conv norm", 0
n_nci   db "conv inner norm", 0
n_nfb   db "ff2 norm", 0
n_dw    db "depthwise taps", 0
n_nf    db "final norm", 0
m_layer db "block ", 0
m_along db ": fd along g and a random direction, err/|g|", 0

section .bss
alignb 8
mb      resb MB_SIZE
rng     resb RNG_SIZE
hmel    resd NMELS
hlab    resd TB * TU
hloss   resd TB
loss1   resd TB
w0      resq 1                  ; params, host copy
g0      resq 1                  ; gradient
u       resq 1                  ; direction
eps     resq 1
tname   resb 64

section .text

; mb's losses summed, times 1/labels: xmm0
loss:
    sub rsp, 40
    lea rcx, [mb]
    mov edx, FW_LOSS
    call model_fwd
    lea rcx, [hloss]
    mov rdx, [mb+MB_LOSS]
    mov r8d, TB * 4
    call gpu_down
    xorpd xmm0, xmm0
    xor ecx, ecx
.s:
    lea rax, [hloss]
    cvtss2sd xmm1, [rax+rcx*4]
    addsd xmm0, xmm1
    inc ecx
    cmp ecx, TB
    jb .s
    mulsd xmm0, [c_scale]
    add rsp, 40
    ret

; one step's gradient for mb into d_grad
grad:
    sub rsp, 40
    call model_zero
    lea rcx, [mb]
    mov edx, FW_GRAD
    call model_fwd
    lea rcx, [mb]
    call model_bwd
    add rsp, 40
    ret

; rcx = device buffer, rdx = host, r8d = bytes: allocate and upload
%macro UP 3
    mov ecx, %3
    call gpu_alloc
    mov [mb+%1], rax
    mov rcx, rax
    lea rdx, [%2]
    mov r8d, %3
    call gpu_up
%endmacro

; the next tensor: count, name, block (-1 = none). r12 = its offset, advanced
%macro CHK 3
    mov rcx, r12
    mov rdx, %1
    lea r8, [%2]
    mov r9, %3
    add r12, rdx
    call fdcheck
%endmacro

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
    xor ecx, ecx
    call pool_init
    mov qword [mdl+MD_L], TL
    mov qword [mdl+MD_D], TD
    mov qword [mdl+MD_H], TH
    mov qword [mdl+MD_F], TF
    mov qword [mdl+MD_V], TV
    mov qword [mdl+MD_K], TK
    mov qword [mdl+MD_C0], TC0
    mov qword [mdl+MD_NMEL], TMEL
    mov qword [mdl+MD_T], TT
    mov qword [mdl+MD_B], TB
    mov qword [mdl+MD_UMAX], TU
    mov rax, __?float64?__(10000.0)
    mov [mdl+MD_ROPE], rax
    mov ecx, 4
    call model_setup
    call model_summary
    mov ecx, 7
    call model_init

    ; random mel (normal), labels (not the blank), on the gpu
    lea rcx, [rng]
    mov edx, 99
    call rng_seed
    xor ebx, ebx
.mel:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    lea rax, [hmel]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NMELS
    jb .mel
    xor ebx, ebx
.lab:
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    mov ecx, TV - 1
    div rcx
    lea rax, [hlab]
    mov [rax+rbx*4], edx
    inc ebx
    cmp ebx, TB * TU
    jb .lab
    UP MB_MEL, hmel, NMELS * 4
    UP MB_LAB, hlab, TB * TU * 4
    UP MB_TL, tls, TB * 4
    UP MB_UL, uls, TB * 4
    mov ecx, TB * 4
    call gpu_alloc
    mov [mb+MB_LOSS], rax
    mov dword [mb+MB_B], TB
    mov dword [mb+MB_T], TT
    mov dword [mb+MB_UMAX], TU
    mov dword [mb+MB_CS], 2     ; chunks of 2 encoder frames, one chunk back
    mov dword [mb+MB_LEFT], 1
    mov eax, __?float32?__(0.16666666666666666)
    mov [mb+MB_SCALE], eax

    call loss
    movsd [rsp+32], xmm0
    say "  loss at init "
    movsd xmm0, [rsp+32]
    mov edx, 5
    call print_fixed
    say 13, 10
    lea rcx, [loss1]
    lea rdx, [hloss]
    mov eax, [rdx]
    mov [rcx], eax
    mov eax, [rdx+4]
    mov [rcx+4], eax
    mov eax, [rdx+8]
    mov [rcx+8], eax

    ; the last utterance has 2 encoder frames, which see mel frames 0..8 at most.
    ; frames 9 on are padding: make them wild, the losses must not move a bit
    mov ebx, (2 * TT + 9) * TMEL
.wild:
    lea rax, [hmel]
    movsd xmm0, [c_wild]
    cvtsd2ss xmm0, xmm0
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, NMELS
    jb .wild
    mov rcx, [mb+MB_MEL]
    lea rdx, [hmel]
    mov r8d, NMELS * 4
    call gpu_up
    call loss
    lea rcx, [loss1]
    lea rdx, [hloss]
    mov eax, [rcx]
    xor eax, [rdx]
    mov r8d, [rcx+4]
    xor r8d, [rdx+4]
    or eax, r8d
    mov r8d, [rcx+8]
    xor r8d, [rdx+8]
    or eax, r8d
    test eax, eax
    check z, "padding doesn't leak: wild mel past the end leaves every loss bit for bit"

    ; the gradient, and a copy of the weights to nudge
    call grad
    mov rbx, [mdl+MD_NP]
    lea rcx, [rbx*4]
    call mem_alloc
    mov [g0], rax
    lea rcx, [rbx*4]
    call mem_alloc
    mov [w0], rax
    lea rcx, [rbx*4]
    call mem_alloc
    mov [u], rax
    mov rcx, [g0]
    mov rdx, [d_grad]
    lea r8, [rbx*4]
    call gpu_down
    mov rcx, [w0]
    mov rdx, [d_params]
    lea r8, [rbx*4]
    call gpu_down

    ; every tensor, in the order they sit in the params
    xor r12d, r12d
    CHK TD * TF3 * TC0, n_wl, -1
    CHK TC0 * TC0, n_wp2, -1
    CHK TC0 * TC0, n_wp3, -1
    xor r13d, r13d
.blk:
    CHK 2 * TF * TD, n_13a, r13
    CHK TF * TD, n_2a, r13
    CHK 3 * TD * TD, n_qkv, r13
    CHK TD * TD, n_o, r13
    CHK 2 * TD * TD, n_pw1, r13
    CHK TD * TD, n_pw2, r13
    CHK 2 * TF * TD, n_13b, r13
    CHK TF * TD, n_2b, r13
    inc r13
    cmp r13, TL
    jb .blk
    CHK TV * TD, n_head, -1
    CHK 10 * TC0, n_w1, -1
    CHK 10 * TC0, n_w2, -1
    CHK 10 * TC0, n_w3, -1
    xor r13d, r13d
.small:
    CHK TD, n_nfa, r13
    CHK TD, n_nat, r13
    CHK TD, n_ncv, r13
    CHK TD, n_nci, r13
    CHK TD, n_nfb, r13
    CHK (TK + 1) * TD, n_dw, r13
    inc r13
    cmp r13, TL
    jb .small
    CHK TD, n_nf, -1
    cmp r12, [mdl+MD_NP]
    check e, "  ...and that was all of them"

    ; overfit the one batch
    call loss
    movsd [rsp+32], xmm0
    mov ebx, 1
.step:
    call grad
    lea rcx, [op]
    mov edx, ebx
    call model_step
    inc ebx
    cmp ebx, 40
    jbe .step
    call loss
    movsd [rsp+40], xmm0
    say "  40 adamw steps on the one batch: loss "
    movsd xmm0, [rsp+32]
    mov edx, 4
    call print_fixed
    say " -> "
    movsd xmm0, [rsp+40]
    mov edx, 4
    call print_fixed
    say 13, 10
    movsd xmm0, [rsp+40]
    mulsd xmm0, [c_four]
    comisd xmm0, [rsp+32]
    check b, "  it fell to under a quarter"
    jmp t_done

section .rdata
align 8
c_four  dq 4.0
section .text

; rcx = offset, rdx = count, r8 = name, r9 = block (-1 = none).
; fd along the gradient and along a random direction, both against g.u
fdcheck:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 64
    mov rsi, rcx
    mov rdi, rdx
    mov r12, r8
    mov r13, r9
    ; |g|
    mov rax, [g0]
    lea rbx, [rax+rsi*4]
    xorpd xmm0, xmm0
    xor ecx, ecx
.n:
    cmp rcx, rdi
    jae .nd
    cvtss2sd xmm1, [rbx+rcx*4]
    mulsd xmm1, xmm1
    addsd xmm0, xmm1
    inc rcx
    jmp .n
.nd:
    sqrtsd xmm0, xmm0
    movsd [rsp+32], xmm0        ; |g|
    ; the step: big enough that the loss moves ~1e-3, at least 1e-3
    movsd xmm1, [c_shift]
    divsd xmm1, xmm0
    maxsd xmm1, [c_eps]
    movsd [eps], xmm1
    ; along g: u = g/|g|, so g.u = |g|
    mov rax, [u]
    xor ecx, ecx
.u1:
    cmp rcx, rdi
    jae .u1d
    cvtss2sd xmm1, [rbx+rcx*4]
    divsd xmm1, [rsp+32]
    cvtsd2ss xmm1, xmm1
    movss [rax+rcx*4], xmm1
    inc rcx
    jmp .u1
.u1d:
    mov rcx, rsi
    mov rdx, rdi
    call fd
    subsd xmm0, [rsp+32]
    call absdiv
    movsd [rsp+40], xmm0
    ; random unit direction
    xorpd xmm0, xmm0
    movsd [rsp+48], xmm0
    xor ebx, ebx
.u2:
    cmp rbx, rdi
    jae .u2d
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    mov rax, [u]
    movss [rax+rbx*4], xmm0
    cvtss2sd xmm0, xmm0
    mulsd xmm0, xmm0
    addsd xmm0, [rsp+48]
    movsd [rsp+48], xmm0
    inc rbx
    jmp .u2
.u2d:
    movsd xmm2, [rsp+48]
    sqrtsd xmm2, xmm2
    mov rax, [u]
    mov rdx, [g0]
    lea rdx, [rdx+rsi*4]
    xorpd xmm3, xmm3            ; g.u
    xor ecx, ecx
.un:
    cmp rcx, rdi
    jae .und
    cvtss2sd xmm1, [rax+rcx*4]
    divsd xmm1, xmm2
    cvtsd2ss xmm1, xmm1
    movss [rax+rcx*4], xmm1
    cvtss2sd xmm1, xmm1
    cvtss2sd xmm4, [rdx+rcx*4]
    mulsd xmm1, xmm4
    addsd xmm3, xmm1
    inc rcx
    jmp .un
.und:
    movsd [rsp+48], xmm3
    mov rcx, rsi
    mov rdx, rdi
    call fd
    subsd xmm0, [rsp+48]
    call absdiv
    ucomisd xmm0, xmm0          ; a nan in either one wins
    jp .worse
    movsd xmm1, [rsp+40]
    ucomisd xmm1, xmm1
    jp .named
    maxsd xmm0, xmm1
.worse:
    movsd [rsp+40], xmm0
.named:
    ; "block 1 wqkv: ..."
    lea rcx, [tname]
    test r13, r13
    js .nolayer
    lea rdx, [m_layer]
    call fmt_str
    mov rcx, rax
    mov rdx, r13
    call fmt_dec
    mov byte [rax], ' '
    lea rcx, [rax+1]
.nolayer:
    mov rdx, r12
    call fmt_str
    mov rcx, rax
    lea rdx, [m_along]
    call fmt_str
    mov byte [rax], 0
    movsd xmm0, [rsp+40]
    comisd xmm0, [c_tol]
    setb cl
    movzx ecx, cl
    lea rdx, [tname]
    movsd xmm2, [rsp+40]
    call t_okf
    add rsp, 64
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; xmm0 = |xmm0| / |g| (the |g| at [rsp+32] of fdcheck's frame, 8 up from here)
absdiv:
    movq rax, xmm0
    btr rax, 63
    movq xmm0, rax
    divsd xmm0, [rsp+8+32]
    ret

; rcx = offset, rdx = count. (L(w + eps u) - L(w - eps u)) / 2 eps
fd:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    movsd xmm0, [eps]
    call nudge
    call loss
    movsd [rsp+32], xmm0
    movsd xmm0, [eps]
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
    call nudge
    call loss
    movsd xmm1, [rsp+32]
    subsd xmm1, xmm0
    movsd xmm0, xmm1
    divsd xmm0, [eps]
    mulsd xmm0, [c_half]
    movsd [rsp+32], xmm0
    xorpd xmm0, xmm0
    call nudge                  ; back to w
    movsd xmm0, [rsp+32]
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rbx = offset, rsi = count (fd's), xmm0 = step. uploads w + step u for that range
nudge:
    push rdi
    sub rsp, 48
    movsd [rsp+32], xmm0
    mov rcx, rsi
    shl rcx, 2
    call mem_alloc
    mov rdi, rax
    mov r8, [w0]
    lea r8, [r8+rbx*4]
    mov r9, [u]
    xor ecx, ecx
.l:
    cmp rcx, rsi
    jae .up
    cvtss2sd xmm0, [r8+rcx*4]
    cvtss2sd xmm1, [r9+rcx*4]
    mulsd xmm1, [rsp+32]
    addsd xmm0, xmm1
    cvtsd2ss xmm0, xmm0
    movss [rdi+rcx*4], xmm0
    inc rcx
    jmp .l
.up:
    mov rcx, [d_params]
    lea rcx, [rcx+rbx*4]
    mov rdx, rdi
    lea r8, [rsi*4]
    call gpu_up
    mov rcx, rdi
    call mem_free
    add rsp, 48
    pop rdi
    ret
