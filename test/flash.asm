; stage 4 test: the flash attention kernels with the encoder's mask, against the
; naive ones (both bf16). T = 100, so the last query block is partial; grouped kv
; heads; a single-frame utterance, so some padding queries see nothing. four masks
; (chunked with left context, chunked alone, full, causal)
; uses: gpu\cuda gpu\kernels
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "gpu/cuda.inc"
%include "test/gpucheck.inc"

extern ptx_attn, ptx_attn_end

NB      equ 3
TT      equ 100
H       equ 4
KVH     equ 2
HD      equ 64
QD      equ H * HD
KVD     equ KVH * HD
QKV     equ QD + 2 * KVD
M       equ NB * TT
SLACK   equ 64                  ; zeroed rows after the last utterance
NS      equ NB * H * TT * TT
NL      equ NB * H * TT

section .rdata
k_dot   db "att_dot", 0
k_sm    db "att_softmax", 0
k_mix   db "att_mix", 0
k_ffwd  db "flash_fwd", 0
k_ds    db "att_ds", 0
k_dkv   db "att_dkv", 0
k_attnd db "attn_d", 0
k_fdq   db "flash_dq", 0
k_fdkv  db "flash_dkv", 0
align 4
lens    dd 100, 70, 1
masks   dd 16, 1
        dd 25, 0
        dd TT, 0
        dd 1, 0xffffffff
align 8
c_scale dq 0.125                ; 1/sqrt(64)
c_one   dq 1.0
c_inf   dd 0x7f800000

section .bss
alignb 8
f_dot   resq 1
f_sm    resq 1
f_mix   resq 1
f_ffwd  resq 1
f_ds    resq 1
f_dkv   resq 1
f_attnd resq 1
f_fdq   resq 1
f_fdkv  resq 1
dyb     resw M * QD
gn      resw M * QKV            ; the qkv's gradient, naive
gf      resw M * QKV            ;   and flash
d_dy    resq 1
d_s2    resq 1
d_D     resq 1
d_gn    resq 1
d_gf    resq 1
rng     resb RNG_SIZE
v_cs    resd 1
v_left  resd 1
qkvb    resw (M + SLACK) * QKV
yn      resw M * QD             ; naive
yf      resw M * QD             ; flash
ln      resd NL
lf      resd NL
back    resd 16
alignb 8
d_qkv   resq 1
d_s1    resq 1
d_lsen  resq 1
d_lsef  resq 1
d_yn    resq 1
d_yf    resq 1
d_len   resq 1

section .text

; the 6 shape args from slot 3 (B, T, H, KVH, hd, bf=1)
shape6:
    mov qword [rbx+KL_ARGS+24], NB
    mov qword [rbx+KL_ARGS+32], TT
    mov qword [rbx+KL_ARGS+40], H
    mov qword [rbx+KL_ARGS+48], KVH
    mov qword [rbx+KL_ARGS+56], HD
    mov qword [rbx+KL_ARGS+64], 1
    ret

; xmm0 (f64) as an f32 kernel arg in slot ecx
fslot:
    cvtsd2ss xmm0, xmm0
    movd eax, xmm0
    mov [rbx+KL_ARGS+rcx*8], rax
    ret

; naive forward: y -> d_yn, lse -> d_lsen
naive:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS], rax
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_s1]
    mov [rbx+KL_ARGS+16], rax
    call shape6
    movsd xmm0, [c_scale]
    mov ecx, 9
    call fslot
    mov qword [rbx+KL_ARGS+80], 0
    mov rax, [d_len]
    mov [rbx+KL_ARGS+88], rax
    mov eax, [v_cs]
    mov [rbx+KL_ARGS+96], rax
    mov eax, [v_left]
    mov [rbx+KL_ARGS+104], rax
    mov ecx, NS
    mov rax, [f_dot]
    call go_flat
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lsen]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], NL
    mov qword [rbx+KL_ARGS+24], TT
    mov qword [rbx+KL_ARGS+32], 0
    mov ecx, NL
    mov rax, [f_sm]
    call go_flat
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_yn]
    mov [rbx+KL_ARGS+16], rax
    call shape6
    movsd xmm0, [c_one]
    mov ecx, 9
    call fslot
    mov qword [rbx+KL_ARGS+80], 0
    mov ecx, M * QD
    mov rax, [f_mix]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; flash forward: y -> d_yf, lse -> d_lsef
flash:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov rax, [f_ffwd]
    mov [rbx+KL_FUNC], rax
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_yf]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_lsef]
    mov [rbx+KL_ARGS+16], rax
    mov qword [rbx+KL_ARGS+24], TT
    mov qword [rbx+KL_ARGS+32], H
    mov qword [rbx+KL_ARGS+40], KVH
    movsd xmm0, [c_scale]
    mov ecx, 6
    call fslot
    mov rax, [d_len]
    mov [rbx+KL_ARGS+56], rax
    mov eax, [v_cs]
    mov [rbx+KL_ARGS+64], rax
    mov eax, [v_left]
    mov [rbx+KL_ARGS+72], rax
    mov dword [rbx+KL_GX], (TT + 63) / 64
    mov dword [rbx+KL_GY], H
    mov dword [rbx+KL_GZ], NB
    mov dword [rbx+KL_BX], 128
    mov dword [rbx+KL_BY], 1
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    add rsp, 32
    pop rbx
    ret

; ecx = mask. both forwards, then y (max diff over the largest) in xmm0 and the
; lse check: eax = 1 if every finite lse is within 1e-2 and every +inf matches
one_mask:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    lea rax, [masks]
    mov edx, [rax+rcx*8]
    mov [v_cs], edx
    mov edx, [rax+rcx*8+4]
    mov [v_left], edx
    ; the outputs start out as garbage, so unwritten rows would show
    mov rcx, [d_yf]
    mov edx, 0x7fc07fc0
    mov r8d, M * QD / 2
    CU cuMemsetD32
    mov rcx, [d_lsef]
    mov edx, 0x7fc00000
    mov r8d, NL
    CU cuMemsetD32
    call naive
    call flash
    call gpu_sync
    lea rcx, [yn]
    mov rdx, [d_yn]
    mov r8d, M * QD * 2
    call gpu_down
    lea rcx, [yf]
    mov rdx, [d_yf]
    mov r8d, M * QD * 2
    call gpu_down
    lea rcx, [ln]
    mov rdx, [d_lsen]
    mov r8d, NL * 4
    call gpu_down
    lea rcx, [lf]
    mov rdx, [d_lsef]
    mov r8d, NL * 4
    call gpu_down
    ; y
    xorpd xmm2, xmm2
    xorpd xmm3, xmm3
    xor r11d, r11d
    xor ecx, ecx
.y:
    lea rax, [yn]
    movzx edx, word [rax+rcx*2]
    shl edx, 16
    movd xmm0, edx
    cvtss2sd xmm0, xmm0
    lea rax, [yf]
    movzx edx, word [rax+rcx*2]
    shl edx, 16
    movd xmm1, edx
    cvtss2sd xmm1, xmm1
    subsd xmm1, xmm0
    ucomisd xmm1, xmm1
    jnp .yn
    inc r11d
.yn:
    andpd xmm1, [t_abs]
    andpd xmm0, [t_abs]
    maxsd xmm2, xmm1
    maxsd xmm3, xmm0
    inc ecx
    cmp ecx, M * QD
    jb .y
    divsd xmm2, xmm3
    movsd [rsp+32], xmm2
    test r11d, r11d
    jz .lse
    movsd xmm0, [t_nan]
    movsd [rsp+32], xmm0
.lse:
    ; lse: +inf where naive has +inf, close elsewhere
    mov ebx, 1
    xor ecx, ecx
.l:
    lea rax, [ln]
    mov edx, [rax+rcx*4]
    lea rax, [lf]
    mov r8d, [rax+rcx*4]
    cmp edx, 0x7f800000
    jne .fin
    cmp r8d, 0x7f800000
    je .ln
    xor ebx, ebx
    jmp .ln
.fin:
    movd xmm0, edx
    movd xmm1, r8d
    subss xmm1, xmm0
    cvtss2sd xmm1, xmm1
    andpd xmm1, [t_abs]
    mov rax, __?float64?__(1e-2)
    movq xmm0, rax
    comisd xmm0, xmm1           ; nan fails too
    ja .ln
    xor ebx, ebx
.ln:
    inc ecx
    cmp ecx, NL
    jb .l
    movsd xmm0, [rsp+32]
    mov eax, ebx
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; naive backward (the forward's s1 holds P after a recompute): gradient -> d_gn
naive_bwd:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    ; P again, from the saved lse
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS], rax
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_s1]
    mov [rbx+KL_ARGS+16], rax
    call shape6
    movsd xmm0, [c_scale]
    mov ecx, 9
    call fslot
    mov qword [rbx+KL_ARGS+80], 0
    call maskargs
    mov ecx, NS
    mov rax, [f_dot]
    call go_flat
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_lsen]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], NL
    mov qword [rbx+KL_ARGS+24], TT
    mov qword [rbx+KL_ARGS+32], 1
    mov ecx, NL
    mov rax, [f_sm]
    call go_flat
    ; dP = dy v, dS
    mov rax, [d_dy]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_s2]
    mov [rbx+KL_ARGS+16], rax
    call shape6
    movsd xmm0, [c_one]
    mov ecx, 9
    call fslot
    mov qword [rbx+KL_ARGS+80], 1
    call maskargs
    mov ecx, NS
    mov rax, [f_dot]
    call go_flat
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_s2]
    mov [rbx+KL_ARGS+8], rax
    mov qword [rbx+KL_ARGS+16], NL
    mov qword [rbx+KL_ARGS+24], TT
    mov ecx, NL
    mov rax, [f_ds]
    call go_flat
    ; dq, then dk dv
    mov rax, [d_s2]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_gn]
    mov [rbx+KL_ARGS+16], rax
    call shape6
    movsd xmm0, [c_scale]
    mov ecx, 9
    call fslot
    mov qword [rbx+KL_ARGS+80], 1
    mov ecx, M * QD
    mov rax, [f_mix]
    call go_flat
    mov rax, [d_s1]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_s2]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_dy]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_gn]
    mov [rbx+KL_ARGS+32], rax
    mov qword [rbx+KL_ARGS+40], NB
    mov qword [rbx+KL_ARGS+48], TT
    mov qword [rbx+KL_ARGS+56], H
    mov qword [rbx+KL_ARGS+64], KVH
    mov qword [rbx+KL_ARGS+72], HD
    mov qword [rbx+KL_ARGS+80], 1
    movsd xmm0, [c_scale]
    mov ecx, 11
    call fslot
    mov ecx, M * KVD
    mov rax, [f_dkv]
    call go_flat
    add rsp, 32
    pop rbx
    ret

; att_dot's mask slots 11..13
maskargs:
    mov rax, [d_len]
    mov [rbx+KL_ARGS+88], rax
    mov eax, [v_cs]
    mov [rbx+KL_ARGS+96], rax
    mov eax, [v_left]
    mov [rbx+KL_ARGS+104], rax
    ret

; flash backward: D, then dq and dk dv -> d_gf
flash_bwd:
    push rbx
    sub rsp, 32
    lea rbx, [kl]
    mov rax, [d_dy]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_yf]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_D]
    mov [rbx+KL_ARGS+16], rax
    mov qword [rbx+KL_ARGS+24], M
    mov qword [rbx+KL_ARGS+32], TT
    mov qword [rbx+KL_ARGS+40], H
    mov ecx, M * H
    mov rax, [f_attnd]
    call go_flat
    mov rax, [d_qkv]
    mov [rbx+KL_ARGS], rax
    mov rax, [d_dy]
    mov [rbx+KL_ARGS+8], rax
    mov rax, [d_lsef]
    mov [rbx+KL_ARGS+16], rax
    mov rax, [d_D]
    mov [rbx+KL_ARGS+24], rax
    mov rax, [d_gf]
    mov [rbx+KL_ARGS+32], rax
    mov qword [rbx+KL_ARGS+40], TT
    mov qword [rbx+KL_ARGS+48], H
    mov qword [rbx+KL_ARGS+56], KVH
    movsd xmm0, [c_scale]
    mov ecx, 8
    call fslot
    mov rax, [d_len]
    mov [rbx+KL_ARGS+72], rax
    mov eax, [v_cs]
    mov [rbx+KL_ARGS+80], rax
    mov eax, [v_left]
    mov [rbx+KL_ARGS+88], rax
    mov rax, [f_fdq]
    mov [rbx+KL_FUNC], rax
    mov dword [rbx+KL_GX], (TT + 63) / 64
    mov dword [rbx+KL_GY], H
    mov dword [rbx+KL_GZ], NB
    mov dword [rbx+KL_BX], 128
    mov dword [rbx+KL_BY], 1
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    mov rax, [f_fdkv]
    mov [rbx+KL_FUNC], rax
    mov dword [rbx+KL_GY], KVH
    mov rcx, rbx
    call gpu_launch
    add rsp, 32
    pop rbx
    ret

; after one_mask: both backwards, xmm0 = the gradient's max diff over the largest
one_bwd:
    push rbx
    sub rsp, 32
    mov rcx, [d_gf]
    mov edx, 0x7fc07fc0
    mov r8d, M * QKV / 2
    CU cuMemsetD32
    call naive_bwd
    call flash_bwd
    call gpu_sync
    lea rcx, [gn]
    mov rdx, [d_gn]
    mov r8d, M * QKV * 2
    call gpu_down
    lea rcx, [gf]
    mov rdx, [d_gf]
    mov r8d, M * QKV * 2
    call gpu_down
    xorpd xmm2, xmm2
    xorpd xmm3, xmm3
    xor r11d, r11d
    xor ecx, ecx
.g:
    lea rax, [gn]
    movzx edx, word [rax+rcx*2]
    shl edx, 16
    movd xmm0, edx
    cvtss2sd xmm0, xmm0
    lea rax, [gf]
    movzx edx, word [rax+rcx*2]
    shl edx, 16
    movd xmm1, edx
    cvtss2sd xmm1, xmm1
    subsd xmm1, xmm0
    ucomisd xmm1, xmm1
    jnp .gn
    inc r11d
.gn:
    andpd xmm1, [t_abs]
    andpd xmm0, [t_abs]
    maxsd xmm2, xmm1
    maxsd xmm3, xmm0
    inc ecx
    cmp ecx, M * QKV
    jb .g
    divsd xmm2, xmm3
    movapd xmm0, xmm2
    test r11d, r11d
    jz .done
    ; where the first nan is, and how many, to go on
    mov ebx, r11d
    xor ecx, ecx
.f:
    lea rax, [gf]
    movzx edx, word [rax+rcx*2]
    and edx, 0x7fff
    cmp edx, 0x7f80
    ja .found
    inc ecx
    jmp .f
.found:
    mov [rsp+24+32], rcx
    say "    nans: "
    mov ecx, ebx
    call print_dec
    say " first at row "
    mov rax, [rsp+24+32]
    xor edx, edx
    mov ecx, QKV
    div rcx
    mov [rsp+24+32], rdx
    mov rcx, rax
    call print_dec
    say " col "
    mov rcx, [rsp+24+32]
    call print_dec
    say 13, 10
    movsd xmm0, [t_nan]
.done:
    add rsp, 32
    pop rbx
    ret

%macro MASK 2                   ; which, name
    mov ecx, %1
    call one_mask
    mov [rsp+40], eax
    close_to 0.0, 2e-2, %2
    cmp dword [rsp+40], 1
    check e, "    and lse (+inf for queries that see nothing)"
    call one_bwd
    close_to 0.0, 3e-2, "    and the backward: q, k, v gradients"
%endmacro

global start
start:
    push rbx
    sub rsp, 48
    call lib_init
    call gpu_init
    lea rcx, [ptx_attn]
    lea rdx, [ptx_attn_end]
    sub rdx, rcx
    call gpu_module
    mov rbx, rax
    getf rbx, k_dot, f_dot
    getf rbx, k_sm, f_sm
    getf rbx, k_mix, f_mix
    getf rbx, k_ffwd, f_ffwd
    getf rbx, k_ds, f_ds
    getf rbx, k_dkv, f_dkv
    getf rbx, k_attnd, f_attnd
    getf rbx, k_fdq, f_fdq
    getf rbx, k_fdkv, f_fdkv
    lea rcx, [rng]
    mov edx, 21
    call rng_seed
    xor ebx, ebx
.r:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    movd r8d, xmm0
    call bf16
    lea rax, [qkvb]
    mov [rax+rbx*2], r8w
    inc ebx
    cmp ebx, M * QKV
    jb .r
    mov ecx, (M + SLACK) * QKV * 2
    call galloc
    mov [d_qkv], rax
    mov rcx, rax
    lea rdx, [qkvb]
    mov r8d, M * QKV * 2
    call gpu_up
    mov ecx, NS * 4
    call galloc
    mov [d_s1], rax
    mov ecx, NL * 4
    call galloc
    mov [d_lsen], rax
    mov ecx, (NL + SLACK) * 4
    call galloc
    mov [d_lsef], rax
    mov ecx, M * QD * 2
    call galloc
    mov [d_yn], rax
    mov ecx, M * QD * 2
    call galloc
    mov [d_yf], rax
    mov ecx, NB * 4
    call galloc
    mov [d_len], rax
    mov rcx, rax
    lea rdx, [lens]
    mov r8d, NB * 4
    call gpu_up
    ; the backward's: dy random, D and the lse with slack (rows past T read them)
    xor ebx, ebx
.dy:
    lea rcx, [rng]
    call rng_normal
    cvtsd2ss xmm0, xmm0
    movd r8d, xmm0
    call bf16
    lea rax, [dyb]
    mov [rax+rbx*2], r8w
    inc ebx
    cmp ebx, M * QD
    jb .dy
    mov ecx, (M + SLACK) * QD * 2
    call galloc
    mov [d_dy], rax
    mov rcx, rax
    lea rdx, [dyb]
    mov r8d, M * QD * 2
    call gpu_up
    mov ecx, NS * 4
    call galloc
    mov [d_s2], rax
    mov ecx, (NL + SLACK) * 4
    call galloc
    mov [d_D], rax
    mov ecx, M * QKV * 2
    call galloc
    mov [d_gn], rax
    mov ecx, M * QKV * 2
    call galloc
    mov [d_gf], rax

    say "flash forward vs the naive kernels (bf16), y max diff over the largest", 13, 10
    MASK 0, "  chunks of 16, one chunk of left context"
    MASK 1, "  chunks of 25, no left context"
    MASK 2, "  full context"
    MASK 3, "  causal"
    jmp t_done
