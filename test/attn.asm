; stage 4 test: the naive attention kernels with the encoder's mask (utterance
; lengths, chunks of cs frames, `left` chunks of context). a plain f64 version on
; the cpu, its gradient checked by finite differences, then the gpu kernels
; against it under four masks: chunked with left context, chunked alone, full,
; causal. one utterance is a single frame, so some padding queries see nothing
; at all: they must come out 0, not nan
; uses: gpu\cuda gpu\kernels
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "gpu/cuda.inc"
%include "test/gpucheck.inc"

extern ptx_attn, ptx_attn_end

NB      equ 3                   ; utterances
TT      equ 12                  ; frames, padded
H       equ 4                   ; query heads
KVH     equ 2                   ; kv heads, two query heads each
HD      equ 8
QD      equ H * HD
KVD     equ KVH * HD
QKV     equ QD + 2 * KVD
M       equ NB * TT
NS      equ NB * H * TT * TT    ; score matrices

section .rdata
k_dot   db "att_dot", 0
k_sm    db "att_softmax", 0
k_mix   db "att_mix", 0
k_ds    db "att_ds", 0
k_dkv   db "att_dkv", 0
align 4
lens    dd 12, 7, 1
; the masks: chunk size, left chunks
masks   dd 3, 1                 ; chunked, one chunk of context
        dd 4, 0                 ; chunked, none
        dd TT, 0                ; full
        dd 1, 0xffffffff        ; causal
align 8
c_ninf  dq 0xfff0000000000000
c_scale dq 0.35355339059327373  ; 1/sqrt(8)
c_half  dq 0.5
c_eps   dq 1e-5
c_one   dq 1.0

section .bss
alignb 8
f_dot   resq 1
f_sm    resq 1
f_mix   resq 1
f_ds    resq 1
f_dkv   resq 1
rng     resb RNG_SIZE
v_cs    resd 1
v_left  resd 1
v_bwd   resd 1
s_bf    resd 1
alignb 8
; the cpu's, f64
qkv64   resq M * QKV
r64     resq M * QD             ; dy: random on real frames, 0 on padding
y64     resq M * QD
dq64    resq M * QKV
sc      resq TT                 ; one query's scores, then probabilities
dpv     resq TT
okv     resd TT
; for the gpu
qkv32   resd M * QKV
qkvb    resw M * QKV
r32     resd M * QD
rb      resw M * QD
back    resd M * QKV
alignb 8
d_qkv   resq 1
d_qkvb  resq 1
d_r     resq 1
d_rb    resq 1
d_s1    resq 1
d_s2    resq 1
d_lse   resq 1
d_y     resq 1
d_dqkv  resq 1
d_len   resq 1

section .text

; ---- the cpu version, f64. b = rbx, h = r12, t = r13, j = r14 throughout

; eax = 1 if query t may see key j
allowed:
    lea rax, [lens]
    cmp r14d, [rax+rbx*4]
    jae .no
    mov eax, r14d
    xor edx, edx
    div dword [v_cs]
    mov ecx, eax
    mov eax, r13d
    xor edx, edx
    div dword [v_cs]
    cmp ecx, eax
    ja .no
    sub eax, ecx
    cmp eax, [v_left]
    ja .no
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; eax = where query (b, t, h) starts in the qkv
qat:
    imul eax, ebx, TT
    add eax, r13d
    imul eax, eax, QKV
    imul ecx, r12d, HD
    add eax, ecx
    ret

; eax = where key j of h's kv head starts in the qkv (its value is KVD further)
kat:
    mov eax, r12d
    xor edx, edx
    mov ecx, H / KVH
    div ecx
    imul eax, eax, HD
    add eax, QD
    mov ecx, eax
    imul eax, ebx, TT
    add eax, r14d
    imul eax, eax, QKV
    add eax, ecx
    ret

; eax = where (b, t, h) starts in y (and dy)
yat:
    imul eax, ebx, TT
    add eax, r13d
    imul eax, eax, QD
    imul ecx, r12d, HD
    add eax, ecx
    ret

; attention with the mask in v_cs, v_left. y64, and xmm0 = the loss, sum r y.
; with v_bwd set, also dq64 (the gradient of the whole qkv)
cpu_attn:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    cmp dword [v_bwd], 0
    je .nz
    lea rdi, [dq64]
    mov ecx, M * QKV
    xor eax, eax
    rep stosq
.nz:
    xor ebx, ebx
.b:
    xor r12d, r12d
.h:
    xor r13d, r13d
.t:
    ; the scores this query may see, and their max
    mov rax, [c_ninf]
    mov [rsp+40], rax
    xor r14d, r14d
.s:
    call allowed
    lea rcx, [okv]
    mov [rcx+r14*4], eax
    test eax, eax
    jz .sn
    call qat
    mov esi, eax
    call kat
    mov edi, eax
    xorpd xmm0, xmm0
    xor r15d, r15d
.sd:
    lea rcx, [qkv64]
    lea eax, [rsi+r15]
    movsd xmm1, [rcx+rax*8]
    lea eax, [rdi+r15]
    mulsd xmm1, [rcx+rax*8]
    addsd xmm0, xmm1
    inc r15d
    cmp r15d, HD
    jb .sd
    mulsd xmm0, [c_scale]
    lea rcx, [sc]
    movsd [rcx+r14*8], xmm0
    maxsd xmm0, [rsp+40]
    movsd [rsp+40], xmm0
.sn:
    inc r14d
    cmp r14d, TT
    jb .s
    call yat
    mov esi, eax
    mov rax, [c_ninf]
    cmp rax, [rsp+40]
    jne .some
    ; sees nothing: y = 0, and nothing flows back
    xor r15d, r15d
.z:
    lea rcx, [y64]
    lea eax, [rsi+r15]
    mov qword [rcx+rax*8], 0
    inc r15d
    cmp r15d, HD
    jb .z
    jmp .tn
.some:
    ; p = e^(s - max) / sum, 0 where masked
    xorpd xmm0, xmm0
    movsd [rsp+48], xmm0
    xor r14d, r14d
.e:
    lea rcx, [okv]
    cmp dword [rcx+r14*4], 0
    je .en
    lea rcx, [sc]
    movsd xmm0, [rcx+r14*8]
    subsd xmm0, [rsp+40]
    call math_exp
    lea rcx, [sc]
    movsd [rcx+r14*8], xmm0
    addsd xmm0, [rsp+48]
    movsd [rsp+48], xmm0
.en:
    inc r14d
    cmp r14d, TT
    jb .e
    xor r14d, r14d
.nrm:
    lea rcx, [sc]
    lea rdx, [okv]
    cmp dword [rdx+r14*4], 0
    jne .nk
    mov qword [rcx+r14*8], 0
    jmp .nn
.nk:
    movsd xmm0, [rcx+r14*8]
    divsd xmm0, [rsp+48]
    movsd [rcx+r14*8], xmm0
.nn:
    inc r14d
    cmp r14d, TT
    jb .nrm
    ; y[d] = sum_j p_j v_j[d], and the loss
    xor r15d, r15d
.yd:
    xorpd xmm0, xmm0
    xor r14d, r14d
.yj:
    call kat
    add eax, KVD
    add eax, r15d
    lea rcx, [qkv64]
    movsd xmm1, [rcx+rax*8]
    lea rcx, [sc]
    mulsd xmm1, [rcx+r14*8]
    addsd xmm0, xmm1
    inc r14d
    cmp r14d, TT
    jb .yj
    lea eax, [rsi+r15]
    lea rcx, [y64]
    movsd [rcx+rax*8], xmm0
    lea rcx, [r64]
    mulsd xmm0, [rcx+rax*8]
    addsd xmm0, [rsp+32]
    movsd [rsp+32], xmm0
    inc r15d
    cmp r15d, HD
    jb .yd
    cmp dword [v_bwd], 0
    je .tn
    ; backward: dP_j = dy . v_j, D = sum_j p_j dP_j
    xorpd xmm0, xmm0
    movsd [rsp+56], xmm0
    xor r14d, r14d
.dp:
    call kat
    lea edi, [rax+KVD]
    xorpd xmm0, xmm0
    xor r15d, r15d
.dpd:
    lea eax, [rsi+r15]
    lea rcx, [r64]
    movsd xmm1, [rcx+rax*8]
    lea eax, [rdi+r15]
    lea rcx, [qkv64]
    mulsd xmm1, [rcx+rax*8]
    addsd xmm0, xmm1
    inc r15d
    cmp r15d, HD
    jb .dpd
    lea rcx, [dpv]
    movsd [rcx+r14*8], xmm0
    lea rcx, [sc]
    mulsd xmm0, [rcx+r14*8]
    addsd xmm0, [rsp+56]
    movsd [rsp+56], xmm0
    inc r14d
    cmp r14d, TT
    jb .dp
    ; dS_j = p_j (dP_j - D). dq += scale dS_j k_j, dk_j += scale dS_j q, dv_j += p_j dy
    xor r14d, r14d
.ds:
    lea rcx, [dpv]
    movsd xmm2, [rcx+r14*8]
    subsd xmm2, [rsp+56]
    lea rcx, [sc]
    mulsd xmm2, [rcx+r14*8]
    mulsd xmm2, [c_scale]
    call qat
    mov edi, eax
    call kat
    mov r8d, eax
    xor r15d, r15d
.dsd:
    lea rcx, [qkv64]
    lea rdx, [dq64]
    lea eax, [r8+r15]
    movsd xmm1, [rcx+rax*8]
    mulsd xmm1, xmm2
    lea eax, [rdi+r15]
    addsd xmm1, [rdx+rax*8]
    movsd [rdx+rax*8], xmm1
    movsd xmm1, [rcx+rax*8]
    mulsd xmm1, xmm2
    lea eax, [r8+r15]
    addsd xmm1, [rdx+rax*8]
    movsd [rdx+rax*8], xmm1
    lea eax, [rsi+r15]
    lea r9, [r64]
    movsd xmm1, [r9+rax*8]
    lea r9, [sc]
    mulsd xmm1, [r9+r14*8]
    lea eax, [r8+r15+KVD]
    addsd xmm1, [rdx+rax*8]
    movsd [rdx+rax*8], xmm1
    inc r15d
    cmp r15d, HD
    jb .dsd
    inc r14d
    cmp r14d, TT
    jb .ds
.tn:
    inc r13d
    cmp r13d, TT
    jb .t
    inc r12d
    cmp r12d, H
    jb .h
    inc ebx
    cmp ebx, NB
    jb .b
    movsd xmm0, [rsp+32]
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; edx = qkv index. xmm0 = (L(x + eps) - L(x - eps)) / 2 eps
fd:
    push rsi
    sub rsp, 48
    mov esi, edx
    lea rax, [qkv64]
    movsd xmm0, [rax+rsi*8]
    movsd [rsp+32], xmm0
    addsd xmm0, [c_eps]
    movsd [rax+rsi*8], xmm0
    call cpu_attn
    movsd [rsp+40], xmm0
    movsd xmm0, [rsp+32]
    subsd xmm0, [c_eps]
    lea rax, [qkv64]
    movsd [rax+rsi*8], xmm0
    call cpu_attn
    movsd xmm1, [rsp+40]
    subsd xmm1, xmm0
    mulsd xmm1, [c_half]
    divsd xmm1, [c_eps]
    movsd xmm0, [rsp+32]
    lea rax, [qkv64]
    movsd [rax+rsi*8], xmm0
    movapd xmm0, xmm1
    add rsp, 48
    pop rsi
    ret

; ---- the gpu side

; the shape arguments every att_ kernel but softmax and ds takes, from slot 3 on
shape_args:
    mov qword [rbx+KL_ARGS+24], NB
    mov qword [rbx+KL_ARGS+32], TT
    mov qword [rbx+KL_ARGS+40], H
    mov qword [rbx+KL_ARGS+48], KVH
    mov qword [rbx+KL_ARGS+56], HD
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+64], rax
    ret

; xmm0 = f32 scale into slot ecx
fslot:
    cvtsd2ss xmm0, xmm0
    movd eax, xmm0
    mov [rbx+KL_ARGS+rcx*8], rax
    ret

; att_dot: rcx = a, rdx = qkv, r8 = scores, r9d = mode
run_dot:
    push rbx
    push rsi
    sub rsp, 40
    lea rbx, [kl]
    mov esi, r9d
    mov [rbx+KL_ARGS], rcx
    mov [rbx+KL_ARGS+8], rdx
    mov [rbx+KL_ARGS+16], r8
    call shape_args
    movsd xmm0, [c_scale]
    test esi, esi
    jz .s
    movsd xmm0, [c_one]
.s:
    mov ecx, 9
    call fslot
    mov [rbx+KL_ARGS+80], rsi
    mov rax, [d_len]
    mov [rbx+KL_ARGS+88], rax
    mov eax, [v_cs]
    mov [rbx+KL_ARGS+96], rax
    mov eax, [v_left]
    mov [rbx+KL_ARGS+104], rax
    mov ecx, NS
    mov rax, [f_dot]
    call go_flat
    add rsp, 40
    pop rsi
    pop rbx
    ret

; att_softmax on s1: ecx = use the saved lse
run_sm:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lse]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], NB * H * TT
    mov qword [rbx+KL_ARGS+24], TT
    mov [rbx+KL_ARGS+32], rcx
    mov ecx, NB * H * TT
    mov rax, [f_sm]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; att_mix: rcx = P or dS, rdx = out, r8d = mode
run_mix:
    push rbx
    push rsi
    sub rsp, 40
    lea rbx, [kl]
    mov esi, r8d
    mov [rbx+KL_ARGS], rcx
    mov rax, [d_qkv]
    cmp dword [s_bf], 0
    je .q
    mov rax, [d_qkvb]
.q:
    mov [rbx+KL_ARGS+8], rax
    mov [rbx+KL_ARGS+16], rdx
    call shape_args
    movsd xmm0, [c_one]
    test esi, esi
    jz .s
    movsd xmm0, [c_scale]
.s:
    mov ecx, 9
    call fslot
    mov [rbx+KL_ARGS+80], rsi
    mov ecx, M * QD
    mov rax, [f_mix]
    call go_flat
    add rsp, 40
    pop rsi
    pop rbx
    ret

; forward and backward under the current mask and s_bf
gpu_attn:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, [d_qkv]
    mov rdi, [d_r]
    cmp dword [s_bf], 0
    je .go
    mov rsi, [d_qkvb]
    mov rdi, [d_rb]
.go:
    mov rcx, rsi
    mov rdx, rsi
    mov r8, [d_s1]
    xor r9d, r9d
    call run_dot
    xor ecx, ecx
    call run_sm
    mov rcx, [d_s1]
    mov rdx, [d_y]
    xor r8d, r8d
    call run_mix
    ; backward, the way the model does it: P again from the saved lse
    mov rcx, rsi
    mov rdx, rsi
    mov r8, [d_s1]
    xor r9d, r9d
    call run_dot
    mov ecx, 1
    call run_sm
    mov rcx, rdi
    mov rdx, rsi
    mov r8, [d_s2]
    mov r9d, 1
    call run_dot
    lea rbx, [kl]
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_s2]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], NB * H * TT
    mov qword [rbx+KL_ARGS+24], TT
    mov ecx, NB * H * TT
    mov rax, [f_ds]
    call go_flat
    mov rcx, [d_s2]
    mov rdx, [d_dqkv]
    mov r8d, 1
    call run_mix
    lea rbx, [kl]
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_s2]
    mov [rbx+KL_ARGS+8], rax
    mov [rbx+KL_ARGS+16], rsi
    mov [rbx+KL_ARGS+24], rdi
    mov rax, [d_dqkv]
    mov [rbx+KL_ARGS+32], rax
    mov qword [rbx+KL_ARGS+40], NB
    mov qword [rbx+KL_ARGS+48], TT
    mov qword [rbx+KL_ARGS+56], H
    mov qword [rbx+KL_ARGS+64], KVH
    mov qword [rbx+KL_ARGS+72], HD
    mov eax, [s_bf]
    mov [rbx+KL_ARGS+80], rax
    movsd xmm0, [c_scale]
    mov ecx, 11
    call fslot
    mov ecx, M * KVD
    mov rax, [f_dkv]
    call go_flat
    call gpu_sync
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; esi = which mask. runs cpu and gpu (f32) and compares y and the qkv's gradient
one_mask:
    push rbx
    push rsi
    sub rsp, 40
    mov esi, ecx
    lea rax, [masks]
    mov ecx, [rax+rsi*8]
    mov [v_cs], ecx
    mov ecx, [rax+rsi*8+4]
    mov [v_left], ecx
    mov dword [v_bwd], 1
    call cpu_attn
    mov dword [s_bf], 0
    call gpu_attn
    mov rcx, [d_y]
    lea rdx, [y64]
    mov r8d, M * QD
    xor r9d, r9d
    call against
    movsd [rsp+32], xmm0
    mov rcx, [d_dqkv]
    lea rdx, [dq64]
    mov r8d, M * QKV
    xor r9d, r9d
    call against
    movsd xmm1, [rsp+32]        ; the larger of the two, a nan in either wins
    ucomisd xmm0, xmm0
    jp .done
    ucomisd xmm1, xmm1
    jp .y
    maxsd xmm0, xmm1
    jmp .done
.y:
    movapd xmm0, xmm1
.done:
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
    push r13
    sub rsp, 48
    call lib_init
    call gpu_init
    lea rcx, [ptx_attn]
    lea rdx, [ptx_attn_end]
    sub rdx, rcx
    call gpu_module
    mov r12, rax
    getf r12, k_dot, f_dot
    getf r12, k_sm, f_sm
    getf r12, k_mix, f_mix
    getf r12, k_ds, f_ds
    getf r12, k_dkv, f_dkv

    ; qkv: normal, dy: normal on real frames and 0 on padding, both rounded to bf16
    ; so the f32 and bf16 runs start from the same numbers
    lea rcx, [rng]
    mov edx, 11
    call rng_seed
    xor ebx, ebx
.gq:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    movd r8d, xmm0
    call bf16
    lea rax, [qkvb]
    mov [rax+rbx*2], r8w
    shl r8d, 16
    lea rax, [qkv32]
    mov [rax+rbx*4], r8d
    movd xmm0, r8d
    cvtss2sd xmm0, xmm0
    lea rax, [qkv64]
    movsd [rax+rbx*8], xmm0
    inc ebx
    cmp ebx, M * QKV
    jb .gq
    xor ebx, ebx
.gr:
    lea rcx, [rng]
    call rng_normal
    ; frame t = (i / QD) % TT of utterance (i / QD) / TT: 0 if it's padding
    mov eax, ebx
    xor edx, edx
    mov ecx, QD
    div ecx
    xor edx, edx
    mov ecx, TT
    div ecx
    lea rcx, [lens]
    cmp edx, [rcx+rax*4]
    jb .real
    xorpd xmm0, xmm0
.real:
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
    cmp ebx, M * QD
    jb .gr

    say "the cpu version, f64 (chunks of 3, one chunk of left context)", 13, 10
    mov dword [v_cs], 3
    mov dword [v_left], 1
    mov dword [v_bwd], 1
    call cpu_attn
    mov dword [v_bwd], 0
    xorpd xmm0, xmm0
    movsd [rsp+32], xmm0
    xor r13d, r13d
.fd:
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    mov ecx, M * QKV
    div rcx
    mov esi, edx
    call fd
    lea rax, [dq64]
    subsd xmm0, [rax+rsi*8]
    andpd xmm0, [t_abs]
    lea rcx, [rsp+32]
    call keepmax
    inc r13d
    cmp r13d, 60
    jb .fd
    movsd xmm0, [rsp+32]
    close_to 0.0, 1e-7, "  formula vs finite differences, 60 entries of q, k, v, max diff"

    say "the gpu kernels in f32 vs the cpu: y and the gradient, max diff over the largest", 13, 10
    mov ecx, M * QKV * 4
    call galloc
    mov [d_qkv], rax
    mov ecx, M * QKV * 2
    call galloc
    mov [d_qkvb], rax
    mov ecx, M * QD * 4
    call galloc
    mov [d_r], rax
    mov ecx, M * QD * 2
    call galloc
    mov [d_rb], rax
    mov ecx, NS * 4
    call galloc
    mov [d_s1], rax
    mov ecx, NS * 4
    call galloc
    mov [d_s2], rax
    mov ecx, NB * H * TT * 4
    call galloc
    mov [d_lse], rax
    mov ecx, M * QD * 4
    call galloc
    mov [d_y], rax
    mov ecx, M * QKV * 4
    call galloc
    mov [d_dqkv], rax
    mov ecx, NB * 4
    call galloc
    mov [d_len], rax
    mov rcx, [d_qkv]
    lea rdx, [qkv32]
    mov r8d, M * QKV * 4
    call gpu_up
    mov rcx, [d_qkvb]
    lea rdx, [qkvb]
    mov r8d, M * QKV * 2
    call gpu_up
    mov rcx, [d_r]
    lea rdx, [r32]
    mov r8d, M * QD * 4
    call gpu_up
    mov rcx, [d_rb]
    lea rdx, [rb]
    mov r8d, M * QD * 2
    call gpu_up
    mov rcx, [d_len]
    lea rdx, [lens]
    mov r8d, NB * 4
    call gpu_up
    xor ecx, ecx
    call one_mask
    close_to 0.0, 1e-5, "  chunks of 3, one chunk of left context"
    mov ecx, 1
    call one_mask
    close_to 0.0, 1e-5, "  chunks of 4, no left context"
    mov ecx, 2
    call one_mask
    close_to 0.0, 1e-5, "  full context"
    mov ecx, 3
    call one_mask
    close_to 0.0, 1e-5, "  causal"
    ; the single-frame utterance's padding queries in the chunked mask: frame 6 on
    ; is two chunks past its only frame, so they see nothing. y must be exactly 0
    xor ecx, ecx
    call one_mask
    lea rcx, [back]
    mov rdx, [d_y]
    mov r8d, M * QD * 4
    call gpu_down
    xor eax, eax
    xor ecx, ecx
.zy:
    lea rdx, [back+((2*TT+6)*QD)*4]
    or eax, [rdx+rcx*4]
    inc ecx
    cmp ecx, (TT - 6) * QD
    jb .zy
    test eax, eax
    check z, "  queries that see nothing give exactly 0"

    say "the same in bf16 (chunks of 3, one chunk of left context)", 13, 10
    mov dword [v_cs], 3
    mov dword [v_left], 1
    mov dword [v_bwd], 1
    call cpu_attn
    mov dword [s_bf], 1
    call gpu_attn
    mov rcx, [d_y]
    lea rdx, [y64]
    mov r8d, M * QD
    mov r9d, 1
    call against
    close_to 0.0, 1e-2, "  y"
    mov rcx, [d_dqkv]
    lea rdx, [dq64]
    mov r8d, M * QKV
    mov r9d, 1
    call against
    close_to 0.0, 2e-2, "  the qkv's gradient"
    jmp t_done
