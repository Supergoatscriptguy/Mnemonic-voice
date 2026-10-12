; stage 4 test: the fast path against the reference. one small model in shapes the
; fast kernels take (heads of 64, sizes in 128s), a chunked attention mask, four
; utterances of different lengths. the gradient comes out of the f32 naive path
; (mm_ref, the naive attention) and the bf16 fast one (gemm_tc, flash), from the
; same weights and data; every tensor's two gradients have to point the same way
; (cosine) and be about as big
; uses: gpu\cuda gpu\kernels asr\model
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "asr/model.inc"
%include "test/check.inc"

TL      equ 2
TD      equ 128
TH      equ 2                   ; heads of 64
TF      equ 128
TV      equ 128
TK      equ 9
TC0     equ 128
TMEL    equ 80
TT      equ 256                 ; mel frames: 128, 64, 32 after the convs
TB      equ 4                   ; 4 x 32 = 128 encoder rows
TU      equ 10
TF3     equ 10
NMELS   equ TB * TT * TMEL

section .rdata
align 4
tls     dd 32, 25, 17, 9
uls     dd 10, 8, 5, 3          ; 26 labels
align 8
n_tens  dq TD * TF3 * TC0, TC0 * TC0, TC0 * TC0
c_cos   dq 0.99
c_rel   dq 0.05

section .bss
alignb 8
mb      resb MB_SIZE
rng     resb RNG_SIZE
hmel    resd NMELS
hlab    resd TB * TU
hloss   resd TB
g32     resq 1
g16     resq 1
loss32  resq 1
loss16  resq 1
worstc  resq 1                  ; the lowest cosine
worstr  resq 1                  ; the biggest relative difference of the norms

section .text

; rcx = device buffer field in mb, rdx = host, r8d = bytes
%macro UP 3
    mov ecx, %3
    call gpu_alloc
    mov [mb+%1], rax
    mov rcx, rax
    lea rdx, [%2]
    mov r8d, %3
    call gpu_up
%endmacro

; the model's shape into mdl
shape:
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
    ret

; the gradient for mb into a new host buffer: rax. xmm0 = the loss per label
grad:
    push rbx
    sub rsp, 32
    call model_zero
    lea rcx, [mb]
    mov edx, FW_GRAD
    call model_fwd
    lea rcx, [mb]
    call model_bwd
    mov rcx, [mdl+MD_NP]
    shl rcx, 2
    call mem_alloc
    mov rbx, rax
    mov rcx, rax
    mov rdx, [d_grad]
    mov r8, [mdl+MD_NP]
    shl r8, 2
    call gpu_down
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
    mov eax, 26
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    mov rax, rbx
    add rsp, 32
    pop rbx
    ret

; rcx = offset, rdx = count: cosine and norm ratio of the two gradients there,
; folded into the worst so far
tensor:
    push rbx
    sub rsp, 32
    mov rax, [g32]
    lea r8, [rax+rcx*4]
    mov rax, [g16]
    lea r9, [rax+rcx*4]
    xorpd xmm0, xmm0            ; a.b
    xorpd xmm1, xmm1            ; a.a
    xorpd xmm2, xmm2            ; b.b
    xor eax, eax
.d:
    cmp rax, rdx
    jae .dd
    cvtss2sd xmm3, [r8+rax*4]
    cvtss2sd xmm4, [r9+rax*4]
    movapd xmm5, xmm3
    mulsd xmm5, xmm4
    addsd xmm0, xmm5
    mulsd xmm3, xmm3
    addsd xmm1, xmm3
    mulsd xmm4, xmm4
    addsd xmm2, xmm4
    inc rax
    jmp .d
.dd:
    sqrtsd xmm1, xmm1
    sqrtsd xmm2, xmm2
    movapd xmm3, xmm1
    mulsd xmm3, xmm2
    divsd xmm0, xmm3            ; cosine (nan if a gradient is all 0, which counts as bad)
    minsd xmm0, [worstc]
    ucomisd xmm0, xmm0
    jnp .c
    mov rax, __?float64?__(-1.0)
    movq xmm0, rax
.c:
    movsd [worstc], xmm0
    subsd xmm2, xmm1
    divsd xmm2, xmm1
    mov rax, 0x7fffffffffffffff
    movq xmm3, rax
    andpd xmm2, xmm3
    maxsd xmm2, [worstr]
    movsd [worstr], xmm2
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
    xor ecx, ecx
    call pool_init

    ; the data
    lea rcx, [rng]
    mov edx, 33
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
    mov dword [mb+MB_CS], 8     ; chunks of 8 encoder frames, one back
    mov dword [mb+MB_LEFT], 1
    mov eax, __?float32?__(0.038461538461538464)    ; 1/26
    mov [mb+MB_SCALE], eax

    ; the reference: f32, naive
    call shape
    mov dword [mdl_fast], 0
    mov ecx, 4
    call model_setup
    mov ecx, 7
    call model_init
    call grad
    mov [g32], rax
    movsd [loss32], xmm0
    ; the fast path: bf16, gemm_tc, flash. the same weights from the same seed
    call shape
    mov dword [mdl_fast], 1
    mov ecx, 2
    call model_setup
    mov ecx, 7
    call model_init
    call grad
    mov [g16], rax
    movsd [loss16], xmm0

    say "  loss per label: f32 naive "
    movsd xmm0, [loss32]
    mov edx, 5
    call print_fixed
    say ", bf16 fast "
    movsd xmm0, [loss16]
    mov edx, 5
    call print_fixed
    say 13, 10
    movsd xmm0, [loss16]
    subsd xmm0, [loss32]
    divsd xmm0, [loss32]
    close_to 0.0, 2e-2, "the losses agree (relative)"

    ; every tensor: front end matrices, the blocks' 8, the head, then the small ones
    mov rax, __?float64?__(1.0)
    mov [worstc], rax
    mov qword [worstr], 0
    xor r12d, r12d              ; offset
%macro T 1
    mov rcx, r12
    mov rdx, %1
    add r12, rdx
    call tensor
%endmacro
    T TD * TF3 * TC0
    T TC0 * TC0
    T TC0 * TC0
    xor r13d, r13d
.blk:
    T 2 * TF * TD
    T TF * TD
    T 3 * TD * TD
    T TD * TD
    T 2 * TD * TD
    T TD * TD
    T 2 * TF * TD
    T TF * TD
    inc r13
    cmp r13, TL
    jb .blk
    T TV * TD
    T 10 * TC0
    T 10 * TC0
    T 10 * TC0
    xor r13d, r13d
.small:
    T TD
    T TD
    T TD
    T TD
    T TD
    T (TK + 1) * TD
    inc r13
    cmp r13, TL
    jb .small
    T TD
    cmp r12, [mdl+MD_NP]
    check e, "  every tensor compared"
    movsd xmm0, [worstc]
    movsd xmm1, [c_cos]
    comisd xmm0, xmm1
    seta cl
    movzx ecx, cl
    lea rdx, [m_cos]
    movapd xmm2, xmm0
    call t_okf
    movsd xmm0, [worstr]
    close_to 0.0, 0.05, "  the worst tensor's gradient norms differ by (relative)"
    jmp t_done

section .rdata
m_cos   db "  every tensor's gradients point the same way: the lowest cosine", 0
