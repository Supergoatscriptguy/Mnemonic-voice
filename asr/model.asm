; the speech encoder on the gpu: config, parameter layout, buffers, init, and the
; forward and backward passes as a string of kernel launches. the layout and the
; launch plumbing follow mnemonic's model\model.asm (no fp8, one stream). matmuls
; go to mm_ref or gemm_tc depending on mdl_fast; attention is the naive kernels for
; now (the flash ones don't know the chunk mask yet)
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "asr/model.inc"

extern ptx_basic, ptx_basic_end, ptx_gemm, ptx_gemm_end, ptx_ops, ptx_ops_end
extern ptx_attn, ptx_attn_end, ptx_conv, ptx_conv_end, ptx_sub, ptx_sub_end
extern ptx_ctc, ptx_ctc_end

NCH     equ 64                  ; row chunks for the norm and depthwise weight gradients
SCH     equ 256                 ; output chunks for the subsampling convs' weight gradients
NSQ     equ 256                 ; blocks for the gradient norm

; model_init's work, per tensor
IC_PTR  equ 0
IC_STD  equ 8
IC_SEED equ 16

section .rdata
kn_rms     db "rmsnorm", 0
kn_rmsb    db "rmsnorm_bwd", 0
kn_rmsdw   db "rms_dw", 0
kn_colsum  db "colsum", 0
kn_rope    db "rope", 0
kn_swi     db "swiglu", 0
kn_swib    db "swiglu_bwd", 0
kn_adamw   db "adamw", 0
kn_sumsq   db "sumsq", 0
kn_split   db "splitsum", 0
kn_attdot  db "att_dot", 0
kn_attsm   db "att_softmax", 0
kn_attmix  db "att_mix", 0
kn_attds   db "att_ds", 0
kn_attdkv  db "att_dkv", 0
kn_ffwd    db "flash_fwd", 0
kn_attnd   db "attn_d", 0
kn_fdq     db "flash_dq", 0
kn_fdkv    db "flash_dkv", 0
kn_mmref   db "mm_ref", 0
kn_gemmtc  db "gemm_tc", 0
kn_f2bf    db "f2bf", 0
kn_glu     db "glu", 0
kn_glub    db "glu_bwd", 0
kn_silu    db "silu", 0
kn_silub   db "silu_bwd", 0
kn_dwc     db "dwconv", 0
kn_dwcx    db "dwconv_bwdx", 0
kn_dwcw    db "dwconv_dw", 0
kn_sconv   db "sconv", 0
kn_sconvx  db "sconv_bwdx", 0
kn_sconvw  db "sconv_dw", 0
kn_relu    db "relu", 0
kn_relub   db "relu_bwd", 0
kn_lsm     db "lsm", 0
kn_alpha   db "ctc_alpha", 0
kn_beta    db "ctc_beta", 0
kn_cgrad   db "ctc_grad", 0
kn_argmax  db "argmax", 0

ck_layer   db "n_layer", 0
ck_d       db "d_model", 0
ck_head    db "n_head", 0
ck_ffn     db "ffn", 0
ck_vocab   db "vocab", 0
ck_kernel  db "conv_kernel", 0
ck_c0      db "sub_channels", 0
ck_mel     db "n_mel", 0
ck_frames  db "max_frames", 0
ck_micro   db "micro", 0
ck_labels  db "max_labels", 0
ck_rope    db "rope_base", 0
e_heads    db "d_model has to be a multiple of n_head", 0
e_fast     db "the fast matmuls need bf16, head dim 64, and d_model, ffn, vocab, sub_channels and the front end's last width multiples of 128", 0
e_rows     db "the fast matmuls need a micro-batch with frames a multiple of 8 and utterances * frames/8 a multiple of 128", 0
e_big      db "micro-batch bigger than the model was set up for", 0

align 8
c_rope     dq 10000.0
c_std      dq 0.02
c_one      dq 1.0
c_two      dq 2.0
c_four     dq 4.0
c_six      dq 6.0
c_twelve   dq 12.0
c_onef     dd 1.0

section .bss
alignb 8
global mdl, mdl_fast, lp, fe, d_params, d_bf, d_grad, d_adm, d_adv, d_logits, d_gn
mdl        resb MD_SIZE
fe         resb FE_SIZE
mdl_fast   resd 1
mdl_bf     resd 1               ; 1 if activations are bf16
mmbf       resq 1               ; MM_BF, or 0 when activations are f32
mmbftb     resq 1               ; same plus MM_TB
lp         resq 1               ; host array of per-block pointers (LP_*)
kl         resb KL_SIZE
cur_mb     resq 1               ; the micro-batch being worked on
cur_b      resq 1               ; its utterances
cur_tm     resq 1               ; mel frames
cur_t1     resq 1               ; frames after each stride-2 conv
cur_t2     resq 1
cur_t      resq 1               ; ...the encoder's (T3)
cur_m      resq 1               ; encoder rows, B * T3
cur_r2     resq 1               ; rows of the front end's pointwise matmuls: B*T2*F2
cur_r3     resq 1               ;   and B*T3*F3
scale      resd 1               ; 1/sqrt(hd)
fe_k3      resq 1               ; F3 * C0, the width going into WL
fe_r1max   resq 1               ; utterances * frames the front end's buffers hold
fe_r2max   resq 1               ;   after the first and second convs

d_params   resq 1               ; f32 master weights
d_bf       resq 1               ; bf16 copy for the matmuls
d_grad     resq 1
d_adm      resq 1               ; adam moments
d_adv      resq 1
d_act      resq 1               ; saved activations, L blocks of ACTL and the final one
d_logits   resq 1               ; f32 [M][V]
d_lp       resq 1               ; log probabilities
d_dlog     resq 1               ; the ctc gradient, f32
d_dlogt    resq 1               ; ...as T for the matmuls
d_alpha    resq 1
d_beta     resq 1
d_dres     resq 1               ; gradient of the residual stream, f32
d_drest    resq 1               ; ...and as T
d_dxn      resq 1               ; f32 [M][D]
d_dy       resq 1               ; T [M][D]
d_dqkv     resq 1
d_dh       resq 1               ; T [M][2F]
d_dg       resq 1               ; T [M][F]
d_ds       resq 1               ; conv module, f32 [M][D]: into silu's backward
d_dz       resq 1               ;   out of it
d_dyc      resq 1               ;   the depthwise conv's output gradient
d_dyct     resq 1               ;   (rmsnorm_bwd's T copy of it, not used)
d_dgc      resq 1               ;   T [M][D]
d_dhc      resq 1               ;   T [M][2D]
d_parts    resq 1               ; weight gradient partials
d_sq       resq 1
d_gn       resq 1               ; squared gradient norm
d_split    resq 1               ; split-k partials
d_rope     resq 1
d_s1       resq 1               ; attention scratch, B*H*T3*T3 f32 each
d_s2       resq 1
vram       resq 1               ; bytes we allocated

; kernels
k_rms      resq 1
k_rmsb     resq 1
k_rmsdw    resq 1
k_colsum   resq 1
k_rope     resq 1
k_swi      resq 1
k_swib     resq 1
k_adamw    resq 1
k_sumsq    resq 1
k_splitsum resq 1
k_attdot   resq 1
k_attsm    resq 1
k_attmix   resq 1
k_attds    resq 1
k_attdkv   resq 1
k_ffwd     resq 1
k_attnd    resq 1
k_fdq      resq 1
k_fdkv     resq 1
d_attd     resq 1               ; flash's D: dy . y per query and head
flash      resd 1               ; 1 = attention by the flash kernels (fast, head dim 64)
k_mmref    resq 1
k_gemmtc   resq 1
k_f2bf     resq 1
k_glu      resq 1
k_glub     resq 1
k_silu     resq 1
k_silub    resq 1
k_dwc      resq 1
k_dwcx     resq 1
k_dwcw     resq 1
k_sconv    resq 1
k_sconvx   resq 1
k_sconvw   resq 1
k_relu     resq 1
k_relub    resq 1
k_lsm      resq 1
k_alpha    resq 1
k_beta     resq 1
k_cgrad    resq 1
k_argmax   resq 1

section .text

%macro KF 1
    mov rax, [%1]
    mov [kl+KL_FUNC], rax
%endmacro

; kernel argument n = anything mov rax can take (don't use rax in it)
%macro karg 2
    mov rax, %2
    mov [kl+KL_ARGS+%1*8], rax
%endmacro

%macro kargf 2                  ; f32 from a dword in memory
    mov eax, %2
    mov [kl+KL_ARGS+%1*8], eax
%endmacro

; C = R + A B^T through whichever matmul is active: a, b, c, r, m, n, k, flags
%macro MM 8
    karg 0, %1
    karg 1, %2
    karg 2, %3
    karg 3, %5
    karg 4, %6
    karg 5, %7
    karg 6, %4
    karg 7, %8
    call mm_go
%endmacro

; ecx = threads, 256 per block, 1-D
lin:
    lea eax, [rcx+255]
    shr eax, 8
    mov edx, 1
    mov r8d, 256
    mov r9d, 1
    mov ecx, eax
    ; fall through
; ecx = grid x, edx = grid y, r8d = block x, r9d = block y
launch:
    mov [kl+KL_GX], ecx
    mov [kl+KL_GY], edx
    mov dword [kl+KL_GZ], 1
    mov [kl+KL_BX], r8d
    mov [kl+KL_BY], r9d
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    jmp gpu_launch

; one warp per row, 8 rows a block
rows8:
    lea ecx, [rcx+7]
    shr ecx, 3
    mov edx, 1
    mov r8d, 256
    mov r9d, 1
    jmp launch

; the matmul whose args are in kl
mm_go:
    sub rsp, 56                 ; C, R and the split count live at 32-48
    cmp dword [mdl_fast], 0
    jne .tc
    cmp qword [mdl+MD_ES], 4
    jne .ref
    or dword [kl+KL_ARGS+56], MM_F32
.ref:
    KF k_mmref
    mov ecx, [kl+KL_ARGS+32]    ; N
    add ecx, 15
    shr ecx, 4
    mov edx, [kl+KL_ARGS+24]    ; M
    add edx, 15
    shr edx, 4
    mov r8d, 16
    mov r9d, 16
    call launch
    add rsp, 56
    ret
.tc:
    KF k_gemmtc
    mov ecx, [kl+KL_ARGS+32]
    shr ecx, 7
    mov edx, [kl+KL_ARGS+24]
    shr edx, 7
    ; a weight gradient (A transposed) with too few output tiles to fill the gpu
    ; twice over splits its k (the rows) into up to 8 slices
    test dword [kl+KL_ARGS+56], MM_TA
    jz .one
    mov eax, ecx
    imul eax, edx               ; tiles
    mov r8d, [gpu_nsm]
    add r8d, r8d
    mov r9d, [kl+KL_ARGS+40]
    shr r9d, 5                  ; k tiles
    mov r10d, 1                 ; splits, a power of 2
.more:
    cmp r10d, 8
    jae .split
    mov r11d, eax
    imul r11d, r10d
    cmp r11d, r8d
    jae .split
    lea r11d, [r10d*2-1]
    test r9d, r11d              ; twice as many still has to divide the k tiles
    jnz .split
    add r10d, r10d
    jmp .more
.split:
    cmp r10d, 1
    je .one
    ; the partials go to d_split, then splitsum puts them together with R into C
    mov rax, [kl+KL_ARGS+16]
    mov [rsp+32], rax           ; C
    mov rax, [kl+KL_ARGS+48]
    mov [rsp+40], rax           ; R
    mov rax, [d_split]
    mov [kl+KL_ARGS+16], rax
    mov qword [kl+KL_ARGS+48], 0
    mov [rsp+48], r10
    mov [kl+KL_GX], ecx
    mov [kl+KL_GY], edx
    mov [kl+KL_GZ], r10d
    mov dword [kl+KL_BX], 256
    mov dword [kl+KL_BY], 1
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    call gpu_launch
    mov rcx, [kl+KL_ARGS+24]
    imul rcx, [kl+KL_ARGS+32]   ; M*N
    KF k_splitsum
    karg 0, [rsp+32]
    karg 1, [rsp+40]
    karg 2, [d_split]
    mov [kl+KL_ARGS+24], rcx
    karg 4, [rsp+48]
    cmp ecx, 1 << 20
    jbe .lin
    mov ecx, 1 << 20            ; it strides over the rest
.lin:
    call lin
    add rsp, 56
    ret
.one:
    mov r8d, 256
    mov r9d, 1
    call launch
    add rsp, 56
    ret

%macro cfgi 3                   ; field, key, default
    lea rcx, [%2]
    mov edx, %3
    call cfg_int
    mov [mdl+%1], rax
%endmacro

; the config from the cfg/args, defaults are clio's
global model_config
model_config:
    sub rsp, 40
    cfgi MD_L, ck_layer, 17
    cfgi MD_D, ck_d, 512
    cfgi MD_H, ck_head, 8
    cfgi MD_F, ck_ffn, 1536
    cfgi MD_V, ck_vocab, 1024
    cfgi MD_K, ck_kernel, 9
    cfgi MD_C0, ck_c0, 256
    cfgi MD_NMEL, ck_mel, 80
    cfgi MD_T, ck_frames, 3000
    cfgi MD_B, ck_micro, 16
    cfgi MD_UMAX, ck_labels, 256
    lea rcx, [ck_rope]
    movsd xmm1, [c_rope]
    call cfg_float
    movsd [mdl+MD_ROPE], xmm0
    add rsp, 40
    ret

; rcx = module ptx, rdx = its end, r8 = list of (name, slot) pairs ending in 0
getfuncs:
    push rbx
    push rsi
    sub rsp, 40
    mov rsi, r8
    sub rdx, rcx
    call gpu_module
    mov rbx, rax
.f:
    mov rdx, [rsi]
    test rdx, rdx
    jz .done
    mov rcx, rbx
    call gpu_func
    mov rdx, [rsi+8]
    mov [rdx], rax
    add rsi, 16
    jmp .f
.done:
    add rsp, 40
    pop rsi
    pop rbx
    ret

section .rdata
align 8
fl_ops  dq kn_rms, k_rms, kn_rmsb, k_rmsb, kn_rmsdw, k_rmsdw, kn_colsum, k_colsum
        dq kn_rope, k_rope, kn_swi, k_swi, kn_swib, k_swib, kn_adamw, k_adamw
        dq kn_sumsq, k_sumsq, kn_split, k_splitsum, 0
fl_attn dq kn_attdot, k_attdot, kn_attsm, k_attsm, kn_attmix, k_attmix
        dq kn_attds, k_attds, kn_attdkv, k_attdkv
        dq kn_ffwd, k_ffwd, kn_attnd, k_attnd, kn_fdq, k_fdq, kn_fdkv, k_fdkv, 0
fl_gemm dq kn_mmref, k_mmref, kn_gemmtc, k_gemmtc, 0
fl_basic dq kn_f2bf, k_f2bf, 0
fl_conv dq kn_glu, k_glu, kn_glub, k_glub, kn_silu, k_silu, kn_silub, k_silub
        dq kn_dwc, k_dwc, kn_dwcx, k_dwcx, kn_dwcw, k_dwcw, 0
fl_sub  dq kn_sconv, k_sconv, kn_sconvx, k_sconvx, kn_sconvw, k_sconvw
        dq kn_relu, k_relu, kn_relub, k_relub, 0
fl_ctc  dq kn_lsm, k_lsm, kn_alpha, k_alpha, kn_beta, k_beta, kn_cgrad, k_cgrad
        dq kn_argmax, k_argmax, 0
section .text

; rcx = bytes. device memory, counted in vram
dalloc:
    sub rsp, 40
    add [vram], rcx
    call gpu_alloc
    add rsp, 40
    ret

; round rax up to 256
%macro up256 0
    add rax, 255
    and rax, -256
%endmacro

; rcx = param offset. rax = it as a matmul operand (bf16 copy, or the f32 master on
; the f32 path), rdx = its gradient. keeps rcx
wop:
    mov rax, [d_bf]
    lea rax, [rax+rcx*2]
    cmp qword [mdl+MD_ES], 2
    je .g
    mov rax, [d_params]
    lea rax, [rax+rcx*4]
.g:
    mov rdx, [d_grad]
    lea rdx, [rdx+rcx*4]
    ret

; rcx = param offset. rax = the f32 master, rdx = its gradient. keeps rcx
wsm:
    mov rax, [d_params]
    lea rax, [rax+rcx*4]
    mov rdx, [d_grad]
    lea rdx, [rdx+rcx*4]
    ret

; ecx = bytes per activation (2 = bf16, 4 = f32). works out every size from the
; config in mdl, loads the kernels and allocates everything on the gpu
global model_setup
model_setup:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov [mdl+MD_ES], rcx
    xor eax, eax
    cmp ecx, 2
    sete al
    mov [mdl_bf], eax
    shl eax, 2                  ; MM_BF
    mov [mmbf], rax
    or eax, MM_TB
    mov [mmbftb], rax

    ; shapes
    mov rax, [mdl+MD_D]
    xor edx, edx
    div qword [mdl+MD_H]
    test rdx, rdx
    jnz .badheads
    mov [mdl+MD_HD], rax
    mov rax, [mdl+MD_D]
    lea rax, [rax+rax*2]
    mov [mdl+MD_QKV], rax
    mov rax, [mdl+MD_NMEL]
    inc rax
    shr rax, 1
    mov [mdl+MD_F1], rax
    inc rax
    shr rax, 1
    mov [mdl+MD_F2], rax
    inc rax
    shr rax, 1
    mov [mdl+MD_F3], rax
    imul rax, [mdl+MD_C0]
    mov [fe_k3], rax
    mov rax, [mdl+MD_T]
    add rax, 7
    shr rax, 3                  ; ceil three times over = ceil(T/8)
    mov [mdl+MD_T3], rax
    imul rax, [mdl+MD_B]
    mov [mdl+MD_M], rax
    cmp dword [mdl_fast], 0
    je .shapes
    cmp qword [mdl+MD_HD], 64
    jne .badfast
    cmp qword [mdl+MD_ES], 2
    jne .badfast
    mov eax, 127
    test [mdl+MD_D], rax
    jnz .badfast
    test [mdl+MD_F], rax
    jnz .badfast
    test [mdl+MD_V], rax
    jnz .badfast
    test [mdl+MD_C0], rax
    jnz .badfast
    test [fe_k3], rax
    jnz .badfast
.shapes:
    ; attention by flash on the fast path (it wants bf16 and heads of 64)
    xor eax, eax
    cmp dword [mdl_fast], 0
    je .fl
    cmp qword [mdl+MD_HD], 64
    jne .fl
    mov eax, 1
.fl:
    mov [flash], eax
    cvtsi2sd xmm0, qword [mdl+MD_HD]
    sqrtsd xmm0, xmm0
    movsd xmm1, [c_one]
    divsd xmm1, xmm0
    cvtsd2ss xmm1, xmm1
    movd [scale], xmm1

    ; parameters. matrices first (they get weight decay): WL, WP2, WP3, each block's
    ; eight, WC. then the small ones: W1, W2, W3, each block's five norms and its
    ; depthwise taps, the final norm
    mov rbx, [mdl+MD_D]
    mov rax, [mdl+MD_F]
    imul rax, rbx
    imul rax, 6                 ; 2F + F, twice
    mov rcx, rbx
    imul rcx, rbx
    imul rcx, 7                 ; 3 D + D + 2 D + D
    add rax, rcx
    mov [mdl+MD_LSIZE], rax
    imul rax, [mdl+MD_L]
    mov rcx, [fe_k3]
    imul rcx, rbx
    add rax, rcx
    mov rcx, [mdl+MD_C0]
    imul rcx, rcx
    lea rax, [rax+rcx*2]
    mov rcx, [mdl+MD_V]
    imul rcx, rbx
    add rax, rcx
    mov [mdl+MD_NDEC], rax
    mov rcx, [mdl+MD_C0]
    imul rcx, 30
    add rax, rcx
    mov rcx, [mdl+MD_K]
    add rcx, 6                  ; 5 norms + K+1 rows of taps and bias
    imul rcx, rbx
    imul rcx, [mdl+MD_L]
    add rax, rcx
    add rax, rbx
    mov [mdl+MD_NP], rax

    ; flops per encoder frame: 2 per weight in a matmul forward (the front end's
    ; pointwise ones run 2 F2 + F3 rows per frame), attention at full length (4 T3 D a
    ; block), backward twice the forward
    mov rax, [mdl+MD_LSIZE]
    imul rax, [mdl+MD_L]
    mov rcx, [mdl+MD_V]
    imul rcx, rbx
    add rax, rcx
    mov rcx, [fe_k3]
    imul rcx, rbx
    add rax, rcx
    mov rcx, [mdl+MD_C0]
    imul rcx, rcx
    mov rdx, [mdl+MD_F2]
    add rdx, rdx
    add rdx, [mdl+MD_F3]
    imul rcx, rdx
    add rax, rcx
    mov rcx, [mdl+MD_T3]
    imul rcx, rbx
    imul rcx, [mdl+MD_L]
    add rcx, rcx                ; 4 T3 D L, halved: the 2 below makes it whole
    add rax, rcx
    cvtsi2sd xmm0, rax
    mulsd xmm0, [c_six]
    movsd [mdl+MD_FLOPS], xmm0

    ; kernels
    lea rcx, [ptx_ops]
    lea rdx, [ptx_ops_end]
    lea r8, [fl_ops]
    call getfuncs
    lea rcx, [ptx_attn]
    lea rdx, [ptx_attn_end]
    lea r8, [fl_attn]
    call getfuncs
    lea rcx, [ptx_gemm]
    lea rdx, [ptx_gemm_end]
    lea r8, [fl_gemm]
    call getfuncs
    lea rcx, [ptx_basic]
    lea rdx, [ptx_basic_end]
    lea r8, [fl_basic]
    call getfuncs
    lea rcx, [ptx_conv]
    lea rdx, [ptx_conv_end]
    lea r8, [fl_conv]
    call getfuncs
    lea rcx, [ptx_sub]
    lea rdx, [ptx_sub_end]
    lea r8, [fl_sub]
    call getfuncs
    lea rcx, [ptx_ctc]
    lea rdx, [ptx_ctc_end]
    lea r8, [fl_ctc]
    call getfuncs

    ; parameter buffers
    mov rbx, [mdl+MD_NP]
    lea rcx, [rbx*4]
    call dalloc
    mov [d_params], rax
    lea rcx, [rbx*2]
    call dalloc
    mov [d_bf], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_grad], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_adm], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_adv], rax

    ; activations per block: offsets into a block, r12 = M, r13 = es
    mov r12, [mdl+MD_M]
    mov r13, [mdl+MD_ES]
    xor r14d, r14d
%macro blk 1                    ; slot in offs, rax = bytes
    mov [offs+%1], r14
    up256
    add r14, rax
%endmacro
    mov rbx, r12
    imul rbx, [mdl+MD_D]        ; M*D
    mov r15, rbx
    imul r15, r13               ; M*D*es
    mov rsi, r12
    imul rsi, [mdl+MD_F]
    imul rsi, r13               ; M*F*es
    lea rax, [rbx*4]
    blk LP_X
    mov rax, r15
    blk LP_XNA
    lea rax, [r12*4]
    blk LP_RSA
    lea rax, [rsi*2]
    blk LP_HA
    mov rax, rsi
    blk LP_GA
    lea rax, [rbx*4]
    blk LP_X1
    mov rax, r15
    blk LP_XNT
    lea rax, [r12*4]
    blk LP_RST
    lea rax, [r15+r15*2]
    mov rcx, [mdl+MD_QKV]       ; 64 rows of slack: flash reads past the last utterance
    imul rcx, r13
    shl rcx, 6
    add rax, rcx
    blk LP_QKV
    mov rax, r15
    blk LP_Y
    mov rax, r12
    imul rax, [mdl+MD_H]
    add rax, 64                 ; (and past the last lse)
    shl rax, 2
    blk LP_LSE
    lea rax, [rbx*4]
    blk LP_X2
    mov rax, r15
    blk LP_XNC
    lea rax, [r12*4]
    blk LP_RSC
    lea rax, [r15*2]
    blk LP_HC
    mov rax, r15
    blk LP_GC
    lea rax, [rbx*4]
    blk LP_YC
    mov rax, r15
    blk LP_ZC
    lea rax, [r12*4]
    blk LP_RSZ
    mov rax, r15
    blk LP_SC
    lea rax, [rbx*4]
    blk LP_X3
    mov rax, r15
    blk LP_XNB
    lea rax, [r12*4]
    blk LP_RSB
    lea rax, [rsi*2]
    blk LP_HB
    mov rax, rsi
    blk LP_GB
    mov [mdl+MD_ACTL], r14
    ; L blocks, and the final norm's x, xn, rstd (the first three slots' sizes)
    mov rcx, [mdl+MD_L]
    imul rcx, r14
    add rcx, [offs+LP_HA]
    mov rdi, rcx
    call dalloc
    mov [d_act], rax
    mov rcx, rax                ; zeroed: the slack has to be finite
    xor edx, edx
    mov r8, rdi
    shr r8, 2
    CU cuMemsetD32

    ; the front end's activations and backward scratch, at the most utterances and
    ; frames: a1 [B][T1][F1][C0], a2 [B][T2][F2][C0], a3 [B][T3][F3][C0]
    mov rax, [mdl+MD_T]
    inc rax
    shr rax, 1
    mov rdi, rax                ; T1
    imul rax, [mdl+MD_B]
    mov [fe_r1max], rax         ; utterances * frames these hold
    imul rax, [mdl+MD_F1]
    imul rax, [mdl+MD_C0]
    imul rax, r13
    mov [rsp+32], rax           ; a1 bytes
    mov rax, rdi
    inc rax
    shr rax, 1                  ; T2
    mov rcx, rax
    imul rcx, [mdl+MD_B]
    mov [fe_r2max], rcx
    imul rax, [mdl+MD_F2]
    imul rax, [mdl+MD_B]
    imul rax, [mdl+MD_C0]
    imul rax, r13
    mov [rsp+40], rax           ; a2 bytes
%macro fealloc 2                ; slot, bytes
    mov rcx, %2
    call dalloc
    mov [fe+%1], rax
%endmacro
    fealloc FE_A1, [rsp+32]
    fealloc FE_D1, [rsp+32]
    fealloc FE_B2, [rsp+40]
    fealloc FE_A2, [rsp+40]
    fealloc FE_D2A, [rsp+40]
    fealloc FE_D2B, [rsp+40]
    mov rax, r12
    imul rax, [fe_k3]
    imul rax, r13
    mov [rsp+40], rax           ; a3 bytes, M * F3 * C0 * es
    fealloc FE_B3, [rsp+40]
    fealloc FE_A3, [rsp+40]
    fealloc FE_D3A, [rsp+40]
    fealloc FE_D3B, [rsp+40]

    ; everything that's only needed one micro-batch at a time
    mov rcx, r12
    imul rcx, [mdl+MD_V]
    shl rcx, 2
    mov rdi, rcx
    call dalloc
    mov [d_logits], rax
    mov rcx, rdi
    call dalloc
    mov [d_lp], rax
    mov rcx, rdi
    call dalloc
    mov [d_dlog], rax
    mov [d_dlogt], rax
    cmp qword [mdl+MD_ES], 2
    jne .dlt
    mov rcx, rdi
    shr rcx, 1
    call dalloc
    mov [d_dlogt], rax
.dlt:
    mov rcx, [mdl+MD_UMAX]
    lea rcx, [rcx*2+1]
    imul rcx, r12               ; B*T3 frames, S states each
    shl rcx, 2
    mov rdi, rcx
    call dalloc
    mov [d_alpha], rax
    mov rcx, rdi
    call dalloc
    mov [d_beta], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_dres], rax
    mov rcx, r15
    call dalloc
    mov [d_drest], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_dxn], rax
    mov rcx, [mdl+MD_D]         ; with flash's 64 rows of slack, zeroed
    imul rcx, r13
    shl rcx, 6
    add rcx, r15
    mov rdi, rcx
    call dalloc
    mov [d_dy], rax
    mov rcx, rax
    xor edx, edx
    mov r8, rdi
    shr r8, 2
    CU cuMemsetD32
    mov rcx, r12                ; flash's D, dy . y per query and head, with slack
    imul rcx, [mdl+MD_H]
    add rcx, 64
    shl rcx, 2
    mov rdi, rcx
    call dalloc
    mov [d_attd], rax
    mov rcx, rax
    xor edx, edx
    mov r8, rdi
    shr r8, 2
    CU cuMemsetD32
    lea rcx, [r15+r15*2]
    call dalloc
    mov [d_dqkv], rax
    lea rcx, [rsi*2]
    call dalloc
    mov [d_dh], rax
    mov rcx, rsi
    call dalloc
    mov [d_dg], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_ds], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_dz], rax
    lea rcx, [rbx*4]
    call dalloc
    mov [d_dyc], rax
    mov rcx, r15
    call dalloc
    mov [d_dyct], rax
    mov rcx, r15
    call dalloc
    mov [d_dgc], rax
    lea rcx, [r15*2]
    call dalloc
    mov [d_dhc], rax
    ; partials: the largest of NCH*(K+1)*D (depthwise, also covers the norms) and
    ; SCH*10*C0 (subsampling)
    mov rcx, [mdl+MD_K]
    inc rcx
    imul rcx, [mdl+MD_D]
    imul rcx, NCH
    mov rax, [mdl+MD_C0]
    imul rax, SCH * 10
    cmp rax, rcx
    cmova rcx, rax
    shl rcx, 2
    call dalloc
    mov [d_parts], rax
    mov ecx, NSQ*4
    call dalloc
    mov [d_sq], rax
    mov ecx, 256
    call dalloc
    mov [d_gn], rax
    ; split k only happens under two waves of 128x128 tiles, 8 slices at most
    mov ecx, [gpu_nsm]
    imul ecx, 2 * 8 * 128 * 128 * 4
    call dalloc
    mov [d_split], rax
    mov rcx, [mdl+MD_T3]
    imul rcx, [mdl+MD_HD]
    shl rcx, 2                  ; T3 * hd/2 * (cos, sin) f32
    call dalloc
    mov [d_rope], rax
    mov rcx, r12
    imul rcx, [mdl+MD_H]
    imul rcx, [mdl+MD_T3]
    shl rcx, 2
    mov rdi, rcx
    call dalloc
    mov [d_s1], rax
    mov rcx, rdi
    call dalloc
    mov [d_s2], rax

    ; pointer table, one entry per block plus the final norm
    mov rcx, [mdl+MD_L]
    inc rcx
    imul rcx, LP_SIZE
    call mem_alloc
    mov [lp], rax
    mov rdi, rax
    ; the front end's params
    xor ecx, ecx
    call wop
    mov [fe+FE_WL], rax
    mov [fe+FE_GWL], rdx
    mov rax, [fe_k3]
    imul rax, [mdl+MD_D]
    add rcx, rax
    call wop
    mov [fe+FE_WP2], rax
    mov [fe+FE_GWP2], rdx
    mov rax, [mdl+MD_C0]
    imul rax, rax
    add rcx, rax
    call wop
    mov [fe+FE_WP3], rax
    mov [fe+FE_GWP3], rdx
    mov rax, [mdl+MD_C0]
    imul rax, rax
    add rcx, rax
    mov r12, rcx                ; matrix params so far
    mov rcx, [mdl+MD_NDEC]
    call wsm
    mov [fe+FE_W1], rax
    mov [fe+FE_GW1], rdx
    mov rax, [mdl+MD_C0]
    imul rax, 10
    add rcx, rax
    call wsm
    mov [fe+FE_W2], rax
    mov [fe+FE_GW2], rdx
    mov rax, [mdl+MD_C0]
    imul rax, 10
    add rcx, rax
    call wsm
    mov [fe+FE_W3], rax
    mov [fe+FE_GW3], rdx
    mov rax, [mdl+MD_C0]
    imul rax, 10
    add rcx, rax
    mov r13, rcx                ; small params so far
    mov r14, [d_act]
    xor r15d, r15d              ; block
.lp:
    ; activations (the final entry only uses X, XNA, RSA)
    mov rsi, LP_X
.act:
    lea rax, [offs]
    mov rax, [rax+rsi]
    add rax, r14
    mov [rdi+rsi], rax
    add rsi, 8
    cmp rsi, LP_GB
    jbe .act
    add r14, [mdl+MD_ACTL]
    ; the norms (one for the final entry) and the depthwise taps
    mov rbx, [mdl+MD_D]
    mov rcx, r13
    mov rsi, LP_NFA
.nrm:
    call wsm
    mov [rdi+rsi], rax
    mov [rdi+rsi+LP_GNFA-LP_NFA], rdx
    add rcx, rbx
    cmp r15, [mdl+MD_L]
    je .final
    add rsi, 8
    cmp rsi, LP_NFB
    jbe .nrm
    call wsm
    mov [rdi+LP_DW], rax
    mov [rdi+LP_GDW], rdx
    mov rax, [mdl+MD_K]
    inc rax
    imul rax, rbx
    add rcx, rax
    mov r13, rcx
    ; matrices: ff1 gate+up and down, qkv, o, conv in and out, ff2
    mov rcx, r12
    mov rsi, LP_W13A
    lea rdx, [msz]
    mov rax, [mdl+MD_F]
    add rax, rax
    mov [rdx], rax              ; rows of each, times D
    mov rax, [mdl+MD_F]
    mov [rdx+8], rax
    mov [rdx+56], rax
    add rax, rax
    mov [rdx+48], rax
    mov rax, [mdl+MD_QKV]
    mov [rdx+16], rax
    mov rax, rbx
    mov [rdx+24], rax
    mov [rdx+40], rax
    add rax, rax
    mov [rdx+32], rax
.mat:
    call wop
    mov [rdi+rsi], rax
    mov [rdi+rsi+LP_G13A-LP_W13A], rdx
    lea rax, [msz]
    mov rax, [rax+rsi]
    imul rax, rbx
    add rcx, rax
    add rsi, 8
    cmp rsi, LP_W2B
    jbe .mat
    mov r12, rcx
    add rdi, LP_SIZE
    inc r15
    jmp .lp
.final:
    ; the ctc head, the last matrix
    mov rcx, r12
    call wop
    mov [wc_op], rax
    mov [wc_g], rdx
    call rope_table
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
.badheads:
    lea rcx, [e_heads]
    call fatal
.badfast:
    lea rcx, [e_fast]
    call fatal

section .bss
offs    resq LP_SIZE / 8
msz     resq 8                  ; rows of a block's eight matrices
section .text

; cos/sin table for rope: [t][i] = (cos, sin) of t * base^(-2i/hd), in f64 then f32
rope_table:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    mov rcx, [mdl+MD_T3]
    imul rcx, [mdl+MD_HD]
    shl rcx, 2
    mov r12, rcx
    call mem_alloc
    mov rdi, rax
    xor ebx, ebx                ; t
.t:
    cmp rbx, [mdl+MD_T3]
    jae .up
    xor esi, esi                ; i
.i:
    mov rax, [mdl+MD_HD]
    shr rax, 1
    cmp rsi, rax
    jae .tn
    movsd xmm0, [mdl+MD_ROPE]
    call math_log
    lea rax, [rsi*2]
    cvtsi2sd xmm1, rax
    cvtsi2sd xmm2, qword [mdl+MD_HD]
    divsd xmm1, xmm2
    mulsd xmm0, xmm1
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
    call math_exp               ; base^(-2i/hd)
    cvtsi2sd xmm1, rbx
    mulsd xmm0, xmm1
    movsd [rsp+32], xmm0
    call math_cos
    cvtsd2ss xmm0, xmm0
    mov rax, rbx
    imul rax, [mdl+MD_HD]
    lea rax, [rax+rsi*2]        ; (t*hd/2 + i) * 2 floats
    movss [rdi+rax*4], xmm0
    mov [rsp+40], rax
    movsd xmm0, [rsp+32]
    call math_sin
    cvtsd2ss xmm0, xmm0
    mov rax, [rsp+40]
    movss [rdi+rax*4+4], xmm0
    inc rsi
    jmp .i
.tn:
    inc rbx
    jmp .t
.up:
    mov rcx, [d_rope]
    mov rdx, rdi
    mov r8, r12
    call gpu_up
    mov rcx, rdi
    call mem_free
    add rsp, 56
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = ctx (IC_*), rdx = start, r8 = end. normal(0, std) per element, from a
; stream seeded by the element's index, so it doesn't matter who does which chunk
init_chunk:
    push rbx
    push rsi
    push rdi
    sub rsp, 64
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    lea rcx, [rsp+32]
    mov rdx, [rbx+IC_SEED]
    add rdx, rsi
    call rng_seed
.l:
    cmp rsi, rdi
    jae .done
    lea rcx, [rsp+32]
    call rng_normal
    mulsd xmm0, [rbx+IC_STD]
    cvtsd2ss xmm0, xmm0
    mov rax, [rbx+IC_PTR]
    movss [rax+rsi*4], xmm0
    inc rsi
    jmp .l
.done:
    add rsp, 64
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = host params, rdx = offset, r8 = count, xmm0 = std, r9 = seed
init_range:
    sub rsp, 72
    lea rax, [rcx+rdx*4]
    mov [rsp+32], rax
    movsd [rsp+40], xmm0
    mov rax, 0x9e3779b97f4a7c15
    imul rdx, rax
    add rdx, r9
    mov [rsp+48], rdx
    lea rcx, [init_chunk]
    lea rdx, [rsp+32]
    mov r9d, 65536
    call par_for
    add rsp, 72
    ret

; rbx = host params, r13 = seed, r12 = offset (advanced). r8 = count, xmm0 = std
%macro INIT 0
    mov rcx, rbx
    mov rdx, r12
    add r12, r8
    mov r9, r13
    call init_range
%endmacro

; xmm0 = sqrt(xmm0 / rax)
%macro SQDIV 0
    cvtsi2sd xmm1, rax
    divsd xmm0, xmm1
    sqrtsd xmm0, xmm0
%endmacro

; rcx = seed. fresh weights, then onto the gpu:
;  - block matrices normal(0, 0.02), the four that write into the residual stream
;    (ff downs, o, conv out) scaled down by sqrt(4L), the ctc head 0.02
;  - the front end for relus: 3x3 taps sqrt(2/9), pointwise sqrt(2/C0), WL 1/sqrt(F3 C0)
;  - depthwise taps 1/sqrt(K), every bias 0, every norm 1
global model_init
model_init:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 56
    mov r13, rcx
    mov rcx, [mdl+MD_NP]
    shl rcx, 2
    call mem_alloc
    mov rbx, rax
    mov rdi, rax
    mov rcx, [mdl+MD_NP]
    xor eax, eax
    rep stosd                   ; biases stay 0
    cvtsi2sd xmm0, qword [mdl+MD_L]
    mulsd xmm0, [c_four]
    sqrtsd xmm0, xmm0
    movsd xmm1, [c_std]
    divsd xmm1, xmm0
    movsd [rsp+32], xmm1        ; residual std
    xor r12d, r12d
    ; front end matrices
    movsd xmm0, [c_one]
    mov rax, [fe_k3]
    SQDIV
    mov r8, [fe_k3]
    imul r8, [mdl+MD_D]
    INIT
    movsd xmm0, [c_two]
    mov rax, [mdl+MD_C0]
    SQDIV
    movsd [rsp+40], xmm0
    mov r8, [mdl+MD_C0]
    imul r8, r8
    INIT
    movsd xmm0, [rsp+40]
    mov r8, [mdl+MD_C0]
    imul r8, r8
    INIT
    ; the blocks
    xor esi, esi
.layer:
    cmp rsi, [mdl+MD_L]
    jae .head
    lea r14, [msz]
    xor edi, edi
.mat:
    mov r8, [r14+rdi*8]
    imul r8, [mdl+MD_D]
    movsd xmm0, [c_std]
    cmp edi, 1                  ; ff1 down
    je .res
    cmp edi, 3                  ; o
    je .res
    cmp edi, 5                  ; conv out
    je .res
    cmp edi, 7                  ; ff2 down
    jne .go
.res:
    movsd xmm0, [rsp+32]
.go:
    INIT
    inc edi
    cmp edi, 8
    jb .mat
    inc rsi
    jmp .layer
.head:
    mov r8, [mdl+MD_V]
    imul r8, [mdl+MD_D]
    movsd xmm0, [c_std]
    INIT
    ; W1, W2, W3: nine rows of taps, then the bias row (0)
    xor esi, esi
.sub:
    movsd xmm0, [c_two]
    mov eax, 9
    SQDIV
    mov r8, [mdl+MD_C0]
    imul r8, 9
    INIT
    add r12, [mdl+MD_C0]        ; the bias
    inc esi
    cmp esi, 3
    jb .sub
    ; each block's norms (1) and depthwise taps
    xor esi, esi
.small:
    cmp rsi, [mdl+MD_L]
    jae .fin
    mov rcx, [mdl+MD_D]
    imul rcx, 5
    mov eax, [c_onef]
.one:
    mov [rbx+r12*4], eax
    inc r12
    dec rcx
    jnz .one
    movsd xmm0, [c_one]
    mov rax, [mdl+MD_K]
    SQDIV
    mov r8, [mdl+MD_K]
    imul r8, [mdl+MD_D]
    INIT
    add r12, [mdl+MD_D]         ; the bias
    inc rsi
    jmp .small
.fin:
    mov rcx, [mdl+MD_D]
    mov eax, [c_onef]
.onef:
    mov [rbx+r12*4], eax
    inc r12
    dec rcx
    jnz .onef
    mov rcx, [d_params]
    mov rdx, rbx
    mov r8, [mdl+MD_NP]
    shl r8, 2
    call gpu_up
    mov rcx, rbx
    call mem_free
    call model_adam0            ; adam starts from zero
    call model_cast
    add rsp, 56
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; adam's moments back to zero
global model_adam0
model_adam0:
    sub rsp, 40
    mov rcx, [d_adm]
    xor edx, edx
    mov r8, [mdl+MD_NP]
    CU cuMemsetD32
    mov rcx, [d_adv]
    xor edx, edx
    mov r8, [mdl+MD_NP]
    CU cuMemsetD32
    add rsp, 40
    ret

; bf16 copy of the master weights
global model_cast
model_cast:
    sub rsp, 40
    KF k_f2bf
    karg 0, [d_bf]
    karg 1, [d_params]
    karg 2, [mdl+MD_NP]
    mov ecx, 4096               ; grid-stride
    shl ecx, 8
    call lin
    add rsp, 40
    ret

; gradients back to zero, before a step's first micro-batch
global model_zero
model_zero:
    sub rsp, 40
    mov rcx, [d_grad]
    xor edx, edx
    mov r8, [mdl+MD_NP]
    CU cuMemsetD32
    add rsp, 40
    ret

; ---- launch wrappers

; rcx = x (f32), rdx = w, r8 = out (T), r9 = rstd
rmsnorm:
    sub rsp, 40
    KF k_rms
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    karg 3, r9
    karg 4, [cur_m]
    karg 5, [mdl+MD_D]
    mov eax, [mdl_bf]
    karg 6, rax
    mov rcx, [cur_m]
    call rows8
    add rsp, 40
    ret

; rcx = dy (f32), rdx = x, r8 = rstd, r9 = w, then on the stack: the weight's
; gradient (added to), dx (f32), dxt (dx as T), padd (1 = add into dx)
rmsnorm_bwd:
    push rbx
    sub rsp, 32
    KF k_rmsb
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    karg 3, r9
    karg 4, [rsp+88]
    karg 5, [rsp+96]
    karg 6, [cur_m]
    karg 7, [mdl+MD_D]
    mov eax, [mdl_bf]
    karg 8, rax
    karg 9, [rsp+104]
    mov rcx, [cur_m]
    call rows8
    ; weight gradient: partial sums per row chunk, then add them up.
    ; dy, x, rstd are already in args 0-2
    KF k_rmsdw
    karg 3, [d_parts]
    karg 4, [cur_m]
    karg 5, [mdl+MD_D]
    mov rcx, [mdl+MD_D]
    add ecx, 31
    shr ecx, 5
    mov edx, NCH
    mov r8d, 32
    mov r9d, 8
    call launch
    KF k_colsum
    karg 0, [d_parts]
    karg 1, [rsp+80]
    karg 2, [mdl+MD_D]
    karg 3, NCH
    mov rcx, [mdl+MD_D]
    call lin
    add rsp, 32
    pop rbx
    ret

; call rmsnorm_bwd: dy, x, rstd, w, gw, dx, dxt, padd. needs 64 bytes of frame
%macro RMSB 8
    mov rax, %5
    mov [rsp+32], rax
    mov rax, %6
    mov [rsp+40], rax
    mov rax, %7
    mov [rsp+48], rax
    mov qword [rsp+56], %8
    mov rcx, %1
    mov rdx, %2
    mov r8, %3
    mov r9, %4
    call rmsnorm_bwd
%endmacro

; rcx = qkv (or dqkv), edx = 1 for the backward rotation
rope:
    sub rsp, 40
    mov [rsp+32], rdx
    KF k_rope
    karg 0, rcx
    karg 1, [d_rope]
    karg 2, [cur_m]
    karg 3, [cur_t]
    karg 4, [mdl+MD_HD]
    mov rax, [mdl+MD_H]
    add rax, rax                ; q heads and k heads
    mov [kl+KL_ARGS+40], rax
    karg 6, [mdl+MD_QKV]
    mov eax, [mdl_bf]
    karg 7, rax
    karg 8, [rsp+32]
    mov rcx, [cur_m]
    imul rcx, [kl+KL_ARGS+40]
    mov rax, [mdl+MD_HD]
    shr rax, 1
    imul rcx, rax
    call lin
    add rsp, 40
    ret

; rcx = h, rdx = g, r8 = which (k_swi or k_glu), r9 = width (F or D)
gate:
    sub rsp, 40
    mov rax, [r8]
    mov [kl+KL_FUNC], rax
    karg 0, rcx
    karg 1, rdx
    karg 2, [cur_m]
    mov [kl+KL_ARGS+24], r9
    mov eax, [mdl_bf]
    karg 4, rax
    mov rcx, [cur_m]
    imul rcx, r9
    call lin
    add rsp, 40
    ret

; rcx = h, rdx = dg, r8 = dh, r9 = which (k_swib or k_glub), [rsp+40] = width
gate_bwd:
    sub rsp, 40
    mov rax, [r9]
    mov [kl+KL_FUNC], rax
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    karg 3, [cur_m]
    karg 4, [rsp+80]
    mov eax, [mdl_bf]
    karg 5, rax
    mov rcx, [cur_m]
    imul rcx, [rsp+80]
    call lin
    add rsp, 40
    ret

; B, T3, H, H (as many kv heads), hd, bf into slots 3..8
att_args:
    karg 3, [cur_b]
    karg 4, [cur_t]
    karg 5, [mdl+MD_H]
    karg 6, [mdl+MD_H]
    karg 7, [mdl+MD_HD]
    mov eax, [mdl_bf]
    karg 8, rax
    ret

; the mask from the micro-batch into att_dot's slots 11..13
mask_args:
    mov rax, [cur_mb]
    mov rcx, [rax+MB_TL]
    mov [kl+KL_ARGS+88], rcx
    mov ecx, [rax+MB_CS]
    mov [kl+KL_ARGS+96], rcx
    mov ecx, [rax+MB_LEFT]
    mov [kl+KL_ARGS+104], rcx
    ret

; B*H*T*T, the size of the score matrices
%macro nscores 0
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_H]
    imul rcx, [cur_t]
%endmacro

; rsi = block. P = softmax(scores) into s1 (puse = edx: 1 = from the saved lse)
att_p:
    sub rsp, 40
    mov [rsp+32], rdx
    KF k_attdot
    mov rax, [rsi+LP_QKV]
    karg 0, rax
    karg 1, rax
    karg 2, [d_s1]
    call att_args
    kargf 9, [scale]
    karg 10, 0
    call mask_args
    nscores
    call lin
    KF k_attsm
    karg 0, [d_s1]
    karg 1, [rsi+LP_LSE]
    mov rax, [cur_m]
    imul rax, [mdl+MD_H]
    mov [kl+KL_ARGS+16], rax
    karg 3, [cur_t]
    karg 4, [rsp+32]
    mov rcx, [kl+KL_ARGS+16]
    call lin
    add rsp, 40
    ret

; the flash kernels' trailing args (the mask) and grid (ceil(T/64), ecx heads, B)
flash_go:
    mov rax, [cur_mb]
    mov rdx, [rax+MB_TL]
    mov r8d, [rax+MB_CS]
    mov r9d, [rax+MB_LEFT]
    mov eax, [kl+KL_ARGS+120]   ; (the slot the caller says the mask starts at)
    lea r10, [kl+KL_ARGS]
    mov [r10+rax*8], rdx
    mov [r10+rax*8+8], r8
    mov [r10+rax*8+16], r9
    mov eax, [cur_t]
    add eax, 63
    shr eax, 6
    mov [kl+KL_GX], eax
    mov [kl+KL_GY], ecx
    mov eax, [cur_b]
    mov [kl+KL_GZ], eax
    mov dword [kl+KL_BX], 128
    mov dword [kl+KL_BY], 1
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    jmp gpu_launch

; rsi = block. y and lse from the qkv
attn_fwd:
    sub rsp, 40
    cmp dword [flash], 0
    je .naive
    KF k_ffwd
    karg 0, [rsi+LP_QKV]
    karg 1, [rsi+LP_Y]
    karg 2, [rsi+LP_LSE]
    karg 3, [cur_t]
    karg 4, [mdl+MD_H]
    karg 5, [mdl+MD_H]
    kargf 6, [scale]
    mov qword [kl+KL_ARGS+120], 7
    mov ecx, [mdl+MD_H]
    call flash_go
    add rsp, 40
    ret
.naive:
    xor edx, edx
    call att_p
    KF k_attmix
    karg 0, [d_s1]
    karg 1, [rsi+LP_QKV]
    karg 2, [rsi+LP_Y]
    call att_args
    kargf 9, [c_onef]
    karg 10, 0
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_D]
    call lin
    add rsp, 40
    ret

; rsi = block. dqkv from dy
attn_bwd:
    sub rsp, 40
    cmp dword [flash], 0
    je .naive
    KF k_attnd
    karg 0, [d_dy]
    karg 1, [rsi+LP_Y]
    karg 2, [d_attd]
    karg 3, [cur_m]
    karg 4, [cur_t]
    karg 5, [mdl+MD_H]
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_H]
    call lin
    ; dq, then dk and dv: the same arguments
    KF k_fdq
    karg 0, [rsi+LP_QKV]
    karg 1, [d_dy]
    karg 2, [rsi+LP_LSE]
    karg 3, [d_attd]
    karg 4, [d_dqkv]
    karg 5, [cur_t]
    karg 6, [mdl+MD_H]
    karg 7, [mdl+MD_H]
    kargf 8, [scale]
    mov qword [kl+KL_ARGS+120], 9
    mov ecx, [mdl+MD_H]
    call flash_go
    KF k_fdkv
    mov ecx, [mdl+MD_H]
    call flash_go
    add rsp, 40
    ret
.naive:
    mov edx, 1
    call att_p
    ; dP = dy v, then dS
    KF k_attdot
    karg 0, [d_dy]
    karg 1, [rsi+LP_QKV]
    karg 2, [d_s2]
    call att_args
    kargf 9, [c_onef]
    karg 10, 1
    call mask_args
    nscores
    call lin
    KF k_attds
    karg 0, [d_s1]
    karg 1, [d_s2]
    mov rax, [cur_m]
    imul rax, [mdl+MD_H]
    mov [kl+KL_ARGS+16], rax
    karg 3, [cur_t]
    mov rcx, [kl+KL_ARGS+16]
    call lin
    ; dq
    KF k_attmix
    karg 0, [d_s2]
    karg 1, [rsi+LP_QKV]
    karg 2, [d_dqkv]
    call att_args
    kargf 9, [scale]
    karg 10, 1
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_D]
    call lin
    ; dk, dv
    KF k_attdkv
    karg 0, [d_s1]
    karg 1, [d_s2]
    karg 2, [rsi+LP_QKV]
    karg 3, [d_dy]
    karg 4, [d_dqkv]
    karg 5, [cur_b]
    karg 6, [cur_t]
    karg 7, [mdl+MD_H]
    karg 8, [mdl+MD_H]
    karg 9, [mdl+MD_HD]
    mov eax, [mdl_bf]
    karg 10, rax
    kargf 11, [scale]
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_D]
    call lin
    add rsp, 40
    ret

; the conv module's middle, forward. rsi = block: gc -> yc (depthwise) -> zc (norm)
; -> sc (silu)
convmid:
    sub rsp, 40
    mov rcx, [rsi+LP_HC]
    mov rdx, [rsi+LP_GC]
    lea r8, [k_glu]
    mov r9, [mdl+MD_D]
    call gate
    KF k_dwc
    karg 0, [rsi+LP_GC]
    karg 1, [rsi+LP_DW]
    karg 2, [rsi+LP_YC]
    call dw_args
    call dw_go
    mov rcx, [rsi+LP_YC]
    mov rdx, [rsi+LP_NCI]
    mov r8, [rsi+LP_ZC]
    mov r9, [rsi+LP_RSZ]
    call rmsnorm
    KF k_silu
    karg 0, [rsi+LP_ZC]
    karg 1, [rsi+LP_SC]
    mov rax, [cur_m]
    imul rax, [mdl+MD_D]
    karg 2, rax
    mov eax, [mdl_bf]
    karg 3, rax
    mov rcx, [kl+KL_ARGS+16]
    call lin
    add rsp, 40
    ret

; the depthwise conv's launch: grid (T3, B), 256 threads across the channels
dw_go:
    mov ecx, [cur_t]
    mov edx, [cur_b]
    mov [kl+KL_GX], ecx
    mov [kl+KL_GY], edx
    mov dword [kl+KL_GZ], 1
    mov dword [kl+KL_BX], 256
    mov dword [kl+KL_BY], 1
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    jmp gpu_launch

; M, T3, D, K, bf into the depthwise conv's slots 3..7
dw_args:
    karg 3, [cur_m]
    karg 4, [cur_t]
    karg 5, [mdl+MD_D]
    karg 6, [mdl+MD_K]
    mov eax, [mdl_bf]
    karg 7, rax
    ret

; one of the front end's convs: rcx = x, rdx = w (f32), r8 = y, r9d = flags (bf is
; added), then frames and bins coming in on the stack
sconv:
    sub rsp, 40
    KF k_sconv
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    karg 3, [cur_b]
    karg 4, [rsp+80]
    karg 5, [rsp+88]
    karg 6, [mdl+MD_C0]
    or r9d, [mdl_bf]
    mov [kl+KL_ARGS+56], r9
    ; grid (ceil(F/2), ceil(T/2), B), threads across the channels
    mov rcx, [rsp+88]
    inc rcx
    shr rcx, 1
    mov rdx, [rsp+80]
    inc rdx
    shr rdx, 1
    call grid3
    add rsp, 40
    ret

; the front end convs' launch: grid (ecx, edx, cur_b), 256 threads a block
grid3:
    mov [kl+KL_GX], ecx
    mov [kl+KL_GY], edx
    mov eax, [cur_b]
    mov [kl+KL_GZ], eax
    mov dword [kl+KL_BX], 256
    mov dword [kl+KL_BY], 1
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    jmp gpu_launch

; its backward: rcx = dy, rdx = x, r8 = w, r9 = dx (0: x is the mel, which needs no
; gradient), then on the stack the weight's gradient (added to), frames and bins
; coming in
sconv_bwd:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r9
    test rdi, rdi
    jz .dw
    KF k_sconvx
    karg 0, rbx
    karg 1, r8
    karg 2, rdi
    karg 3, [cur_b]
    karg 4, [rsp+104]
    karg 5, [rsp+112]
    karg 6, [mdl+MD_C0]
    mov eax, [mdl_bf]
    karg 7, rax
    mov rcx, [rsp+112]          ; grid (F, T, B)
    mov rdx, [rsp+104]
    call grid3
.dw:
    KF k_sconvw
    karg 0, rbx
    karg 1, rsi
    karg 2, [d_parts]
    karg 3, [cur_b]
    karg 4, [rsp+104]
    karg 5, [rsp+112]
    karg 6, [mdl+MD_C0]
    mov eax, [mdl_bf]
    test rdi, rdi
    jnz .fl
    or eax, 2                   ; x is the one-channel mel
.fl:
    karg 7, rax
    mov rcx, [mdl+MD_C0]
    add ecx, 31
    shr ecx, 5
    mov edx, SCH
    mov r8d, 32
    mov r9d, 8
    call launch
    KF k_colsum
    karg 0, [d_parts]
    karg 1, [rsp+96]
    mov rax, [mdl+MD_C0]
    imul rax, 10
    mov [kl+KL_ARGS+16], rax
    karg 3, SCH
    mov rcx, [kl+KL_ARGS+16]
    call lin
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = x, rdx = y, r8 = count (relu, T)
relu:
    sub rsp, 40
    KF k_relu
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    mov eax, [mdl_bf]
    karg 3, rax
    mov rcx, r8
    call lin
    add rsp, 40
    ret

; rcx = y (the relu's output), rdx = dy, r8 = dx, r9 = count
relu_bwd:
    sub rsp, 40
    KF k_relub
    karg 0, rcx
    karg 1, rdx
    karg 2, r8
    karg 3, r9
    mov eax, [mdl_bf]
    karg 4, rax
    mov rcx, r9
    call lin
    add rsp, 40
    ret

; rcx = micro-batch (MB_*). its sizes into cur_*
mbsize:
    mov [cur_mb], rcx
    mov eax, [rcx+MB_B]
    mov [cur_b], rax
    mov eax, [rcx+MB_T]
    mov [cur_tm], rax
    inc rax
    shr rax, 1
    mov [cur_t1], rax
    inc rax
    shr rax, 1
    mov [cur_t2], rax
    mov rdx, rax
    imul rdx, [mdl+MD_F2]
    imul rdx, [cur_b]
    mov [cur_r2], rdx
    inc rax
    shr rax, 1
    mov [cur_t], rax
    imul rax, [cur_b]
    mov [cur_m], rax
    imul rax, [mdl+MD_F3]
    mov [cur_r3], rax
    ret

; rcx = micro-batch, edx = FW_LOGITS / FW_LOSS / FW_GRAD
global model_fwd
model_fwd:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov rbx, rcx
    mov r12d, edx
    call mbsize
    ; it has to fit what model_setup sized: frames an utterance, and the rows at
    ; every stage (many short utterances are as fine as a few long ones)
    mov rax, [cur_tm]
    cmp rax, [mdl+MD_T]
    ja .big
    mov rax, [cur_t1]
    imul rax, [cur_b]
    cmp rax, [fe_r1max]
    ja .big
    mov rax, [cur_t2]
    imul rax, [cur_b]
    cmp rax, [fe_r2max]
    ja .big
    mov rax, [cur_m]
    cmp rax, [mdl+MD_M]
    ja .big
    mov eax, [rbx+MB_UMAX]
    cmp rax, [mdl+MD_UMAX]
    ja .big
    cmp dword [mdl_fast], 0
    je .sized
    test qword [cur_tm], 7
    jnz .rows
    test qword [cur_m], 127
    jnz .rows
.sized:
    ; front end: mel -> a1 -> b2 -> a2 -> b3 -> a3 -> x
    mov rax, [mdl+MD_NMEL]
    mov [rsp+40], rax
    mov rax, [cur_tm]
    mov [rsp+32], rax
    mov rcx, [rbx+MB_MEL]
    mov rdx, [fe+FE_W1]
    mov r8, [fe+FE_A1]
    mov r9d, 6
    call sconv
    mov rax, [cur_t1]
    mov [rsp+32], rax
    mov rax, [mdl+MD_F1]
    mov [rsp+40], rax
    mov rcx, [fe+FE_A1]
    mov rdx, [fe+FE_W2]
    mov r8, [fe+FE_B2]
    xor r9d, r9d
    call sconv
    MM [fe+FE_B2], [fe+FE_WP2], [fe+FE_A2], 0, [cur_r2], [mdl+MD_C0], [mdl+MD_C0], [mmbf]
    mov r8, [cur_r2]
    imul r8, [mdl+MD_C0]
    mov rcx, [fe+FE_A2]
    mov rdx, rcx
    call relu
    mov rax, [cur_t2]
    mov [rsp+32], rax
    mov rax, [mdl+MD_F2]
    mov [rsp+40], rax
    mov rcx, [fe+FE_A2]
    mov rdx, [fe+FE_W3]
    mov r8, [fe+FE_B3]
    xor r9d, r9d
    call sconv
    MM [fe+FE_B3], [fe+FE_WP3], [fe+FE_A3], 0, [cur_r3], [mdl+MD_C0], [mdl+MD_C0], [mmbf]
    mov r8, [cur_r3]
    imul r8, [mdl+MD_C0]
    mov rcx, [fe+FE_A3]
    mov rdx, rcx
    call relu
    mov rsi, [lp]
    MM [fe+FE_A3], [fe+FE_WL], [rsi+LP_X], 0, [cur_m], [mdl+MD_D], [fe_k3], 0

    mov rdi, [mdl+MD_F]
    add rdi, rdi                ; 2F
    xor r13d, r13d
.layer:
    cmp r13, [mdl+MD_L]
    jae .final
    ; ff1: x1 = x + swiglu(norm(x) W13^T) W2^T
    mov rcx, [rsi+LP_X]
    mov rdx, [rsi+LP_NFA]
    mov r8, [rsi+LP_XNA]
    mov r9, [rsi+LP_RSA]
    call rmsnorm
    MM [rsi+LP_XNA], [rsi+LP_W13A], [rsi+LP_HA], 0, [cur_m], rdi, [mdl+MD_D], [mmbf]
    mov rcx, [rsi+LP_HA]
    mov rdx, [rsi+LP_GA]
    lea r8, [k_swi]
    mov r9, [mdl+MD_F]
    call gate
    MM [rsi+LP_GA], [rsi+LP_W2A], [rsi+LP_X1], [rsi+LP_X], [cur_m], [mdl+MD_D], [mdl+MD_F], 0
    ; attention: x2 = x1 + attn(rope(norm(x1) Wqkv^T)) Wo^T
    mov rcx, [rsi+LP_X1]
    mov rdx, [rsi+LP_NAT]
    mov r8, [rsi+LP_XNT]
    mov r9, [rsi+LP_RST]
    call rmsnorm
    MM [rsi+LP_XNT], [rsi+LP_WQKV], [rsi+LP_QKV], 0, [cur_m], [mdl+MD_QKV], [mdl+MD_D], [mmbf]
    mov rcx, [rsi+LP_QKV]
    xor edx, edx
    call rope
    call attn_fwd
    MM [rsi+LP_Y], [rsi+LP_WO], [rsi+LP_X2], [rsi+LP_X1], [cur_m], [mdl+MD_D], [mdl+MD_D], 0
    ; conv: x3 = x2 + silu(norm(dwconv(glu(norm(x2) Wpw1^T)))) Wpw2^T
    mov rcx, [rsi+LP_X2]
    mov rdx, [rsi+LP_NCV]
    mov r8, [rsi+LP_XNC]
    mov r9, [rsi+LP_RSC]
    call rmsnorm
    mov rcx, [mdl+MD_D]
    add rcx, rcx                ; (not rax: MM's kargs go through it)
    MM [rsi+LP_XNC], [rsi+LP_WPW1], [rsi+LP_HC], 0, [cur_m], rcx, [mdl+MD_D], [mmbf]
    call convmid
    MM [rsi+LP_SC], [rsi+LP_WPW2], [rsi+LP_X3], [rsi+LP_X2], [cur_m], [mdl+MD_D], [mdl+MD_D], 0
    ; ff2: x4 = x3 + swiglu(norm(x3) W13^T) W2^T, the next block's x
    mov rcx, [rsi+LP_X3]
    mov rdx, [rsi+LP_NFB]
    mov r8, [rsi+LP_XNB]
    mov r9, [rsi+LP_RSB]
    call rmsnorm
    MM [rsi+LP_XNB], [rsi+LP_W13B], [rsi+LP_HB], 0, [cur_m], rdi, [mdl+MD_D], [mmbf]
    mov rcx, [rsi+LP_HB]
    mov rdx, [rsi+LP_GB]
    lea r8, [k_swi]
    mov r9, [mdl+MD_F]
    call gate
    MM [rsi+LP_GB], [rsi+LP_W2B], [rsi+LP_SIZE+LP_X], [rsi+LP_X3], [cur_m], [mdl+MD_D], [mdl+MD_F], 0
    add rsi, LP_SIZE
    inc r13
    jmp .layer

.final:
    mov rcx, [rsi+LP_X]
    mov rdx, [rsi+LP_NFA]
    mov r8, [rsi+LP_XNA]
    mov r9, [rsi+LP_RSA]
    call rmsnorm
    MM [rsi+LP_XNA], [wc_op], [d_logits], 0, [cur_m], [mdl+MD_V], [mdl+MD_D], 0
    cmp r12d, FW_LOGITS
    je .done
    ; ctc: log probabilities, alpha (and the loss), then beta and the gradient
    KF k_lsm
    karg 0, [d_logits]
    karg 1, [d_lp]
    karg 2, [mdl+MD_V]
    mov rcx, [cur_m]
    mov edx, 1
    mov r8d, 256
    mov r9d, 1
    call launch
    KF k_alpha
    call ctc_args
    karg 4, [d_alpha]
    karg 5, [rbx+MB_LOSS]
    karg 6, [cur_t]
    karg 7, [mdl+MD_V]
    mov eax, [rbx+MB_UMAX]
    karg 8, rax
    mov rax, [mdl+MD_V]
    dec rax
    mov [kl+KL_ARGS+72], rax
    mov rcx, [cur_b]
    mov edx, 1
    mov r8d, 1024
    mov r9d, 1
    call launch
    cmp r12d, FW_GRAD
    jne .done
    KF k_beta
    call ctc_args
    karg 4, [d_beta]
    karg 5, [cur_t]
    karg 6, [mdl+MD_V]
    mov eax, [rbx+MB_UMAX]
    karg 7, rax
    mov rax, [mdl+MD_V]
    dec rax
    mov [kl+KL_ARGS+64], rax
    mov rcx, [cur_b]
    mov edx, 1
    mov r8d, 1024
    mov r9d, 1
    call launch
    KF k_cgrad
    call ctc_args
    karg 4, [d_alpha]
    karg 5, [d_beta]
    karg 6, [rbx+MB_LOSS]
    karg 7, [d_dlog]
    karg 8, [cur_t]
    karg 9, [mdl+MD_V]
    mov eax, [rbx+MB_UMAX]
    karg 10, rax
    mov rax, [mdl+MD_V]
    dec rax
    mov [kl+KL_ARGS+88], rax
    kargf 12, [rbx+MB_SCALE]
    mov rcx, [cur_m]
    mov edx, 1
    mov r8d, 256
    mov r9d, 1
    call launch
    cmp dword [mdl_bf], 0
    je .done
    KF k_f2bf
    karg 0, [d_dlogt]
    karg 1, [d_dlog]
    mov rcx, [cur_m]
    imul rcx, [mdl+MD_V]
    mov [kl+KL_ARGS+16], rcx
    call lin
.done:
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
.big:
    lea rcx, [e_big]
    call fatal
.rows:
    lea rcx, [e_rows]
    call fatal

; rbx = micro-batch: lp, labels, frames, label counts into slots 0..3
ctc_args:
    karg 0, [d_lp]
    karg 1, [rbx+MB_LAB]
    karg 2, [rbx+MB_TL]
    karg 3, [rbx+MB_UL]
    ret

section .bss
wc_op   resq 1                  ; the ctc head as a matmul operand
wc_g    resq 1                  ; and its gradient
section .text

; rcx = micro-batch, after model_fwd with FW_GRAD. adds its gradients into d_grad
global model_bwd
model_bwd:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 72
    mov rbx, rcx
    call mbsize
    mov rax, [lp]
    mov rsi, [mdl+MD_L]
    imul rsi, LP_SIZE
    add rsi, rax                ; the final norm's entry
    mov r14, [mdl+MD_F]
    add r14, r14                ; 2F
    mov rdi, [mdl+MD_D]
    add rdi, rdi                ; 2D

    ; logits = xnf Wc^T
    MM [d_dlogt], [wc_op], [d_dxn], 0, [cur_m], [mdl+MD_D], [mdl+MD_V], MM_TB
    MM [d_dlogt], [rsi+LP_XNA], [wc_g], [wc_g], [mdl+MD_V], [mdl+MD_D], [cur_m], MM_TA|MM_TB
    RMSB [d_dxn], [rsi+LP_X], [rsi+LP_RSA], [rsi+LP_NFA], [rsi+LP_GNFA], [d_dres], [d_drest], 0

    mov r13, [mdl+MD_L]
.layer:
    sub rsi, LP_SIZE
    dec r13
    js .front
    ; ff2
    MM [d_drest], [rsi+LP_W2B], [d_dg], 0, [cur_m], [mdl+MD_F], [mdl+MD_D], [mmbftb]
    MM [d_drest], [rsi+LP_GB], [rsi+LP_G2B], [rsi+LP_G2B], [mdl+MD_D], [mdl+MD_F], [cur_m], MM_TA|MM_TB
    mov rax, [mdl+MD_F]
    mov [rsp+32], rax
    mov rcx, [rsi+LP_HB]
    mov rdx, [d_dg]
    mov r8, [d_dh]
    lea r9, [k_swib]
    call gate_bwd
    MM [d_dh], [rsi+LP_W13B], [d_dxn], 0, [cur_m], [mdl+MD_D], r14, MM_TB
    MM [d_dh], [rsi+LP_XNB], [rsi+LP_G13B], [rsi+LP_G13B], r14, [mdl+MD_D], [cur_m], MM_TA|MM_TB
    RMSB [d_dxn], [rsi+LP_X3], [rsi+LP_RSB], [rsi+LP_NFB], [rsi+LP_GNFB], [d_dres], [d_drest], 1

    ; conv module, back from the out projection to the in projection
    MM [d_drest], [rsi+LP_WPW2], [d_ds], 0, [cur_m], [mdl+MD_D], [mdl+MD_D], MM_TB
    MM [d_drest], [rsi+LP_SC], [rsi+LP_GPW2], [rsi+LP_GPW2], [mdl+MD_D], [mdl+MD_D], [cur_m], MM_TA|MM_TB
    KF k_silub
    karg 0, [rsi+LP_ZC]
    karg 1, [d_ds]
    karg 2, [d_dz]
    mov rax, [cur_m]
    imul rax, [mdl+MD_D]
    karg 3, rax
    mov eax, [mdl_bf]
    karg 4, rax
    mov rcx, [kl+KL_ARGS+24]
    call lin
    RMSB [d_dz], [rsi+LP_YC], [rsi+LP_RSZ], [rsi+LP_NCI], [rsi+LP_GNCI], [d_dyc], [d_dyct], 0
    KF k_dwcx
    karg 0, [d_dyc]
    karg 1, [rsi+LP_DW]
    karg 2, [d_dgc]
    call dw_args
    call dw_go
    KF k_dwcw
    karg 0, [d_dyc]
    karg 1, [rsi+LP_GC]
    karg 2, [d_parts]
    call dw_args
    mov rcx, [mdl+MD_D]
    add ecx, 31
    shr ecx, 5
    mov [kl+KL_GX], ecx
    mov dword [kl+KL_GY], NCH
    mov rax, [mdl+MD_K]
    inc eax
    mov [kl+KL_GZ], eax
    mov dword [kl+KL_BX], 32
    mov dword [kl+KL_BY], 8
    mov dword [kl+KL_BZ], 1
    lea rcx, [kl]
    call gpu_launch
    KF k_colsum
    karg 0, [d_parts]
    karg 1, [rsi+LP_GDW]
    mov rax, [mdl+MD_K]
    inc rax
    imul rax, [mdl+MD_D]
    mov [kl+KL_ARGS+16], rax
    karg 3, NCH
    mov rcx, [kl+KL_ARGS+16]
    call lin
    mov rax, [mdl+MD_D]         ; glu: 2D in, D out
    mov [rsp+32], rax
    mov rcx, [rsi+LP_HC]
    mov rdx, [d_dgc]
    mov r8, [d_dhc]
    lea r9, [k_glub]
    call gate_bwd
    MM [d_dhc], [rsi+LP_WPW1], [d_dxn], 0, [cur_m], [mdl+MD_D], rdi, MM_TB
    MM [d_dhc], [rsi+LP_XNC], [rsi+LP_GPW1], [rsi+LP_GPW1], rdi, [mdl+MD_D], [cur_m], MM_TA|MM_TB
    RMSB [d_dxn], [rsi+LP_X2], [rsi+LP_RSC], [rsi+LP_NCV], [rsi+LP_GNCV], [d_dres], [d_drest], 1

    ; attention
    MM [d_drest], [rsi+LP_WO], [d_dy], 0, [cur_m], [mdl+MD_D], [mdl+MD_D], [mmbftb]
    MM [d_drest], [rsi+LP_Y], [rsi+LP_GO], [rsi+LP_GO], [mdl+MD_D], [mdl+MD_D], [cur_m], MM_TA|MM_TB
    call attn_bwd
    mov rcx, [d_dqkv]
    mov edx, 1
    call rope
    MM [d_dqkv], [rsi+LP_WQKV], [d_dxn], 0, [cur_m], [mdl+MD_D], [mdl+MD_QKV], MM_TB
    MM [d_dqkv], [rsi+LP_XNT], [rsi+LP_GQKV], [rsi+LP_GQKV], [mdl+MD_QKV], [mdl+MD_D], [cur_m], MM_TA|MM_TB
    RMSB [d_dxn], [rsi+LP_X1], [rsi+LP_RST], [rsi+LP_NAT], [rsi+LP_GNAT], [d_dres], [d_drest], 1

    ; ff1
    MM [d_drest], [rsi+LP_W2A], [d_dg], 0, [cur_m], [mdl+MD_F], [mdl+MD_D], [mmbftb]
    MM [d_drest], [rsi+LP_GA], [rsi+LP_G2A], [rsi+LP_G2A], [mdl+MD_D], [mdl+MD_F], [cur_m], MM_TA|MM_TB
    mov rax, [mdl+MD_F]
    mov [rsp+32], rax
    mov rcx, [rsi+LP_HA]
    mov rdx, [d_dg]
    mov r8, [d_dh]
    lea r9, [k_swib]
    call gate_bwd
    MM [d_dh], [rsi+LP_W13A], [d_dxn], 0, [cur_m], [mdl+MD_D], r14, MM_TB
    MM [d_dh], [rsi+LP_XNA], [rsi+LP_G13A], [rsi+LP_G13A], r14, [mdl+MD_D], [cur_m], MM_TA|MM_TB
    RMSB [d_dxn], [rsi+LP_X], [rsi+LP_RSA], [rsi+LP_NFA], [rsi+LP_GNFA], [d_dres], [d_drest], 1
    jmp .layer

.front:
    ; x = a3 WL^T, then back through the front end
    MM [d_drest], [fe+FE_WL], [fe+FE_D3A], 0, [cur_m], [fe_k3], [mdl+MD_D], [mmbftb]
    MM [d_drest], [fe+FE_A3], [fe+FE_GWL], [fe+FE_GWL], [mdl+MD_D], [fe_k3], [cur_m], MM_TA|MM_TB
    mov r9, [cur_r3]
    imul r9, [mdl+MD_C0]
    mov rcx, [fe+FE_A3]
    mov rdx, [fe+FE_D3A]
    mov r8, rdx
    call relu_bwd
    MM [fe+FE_D3A], [fe+FE_WP3], [fe+FE_D3B], 0, [cur_r3], [mdl+MD_C0], [mdl+MD_C0], [mmbftb]
    MM [fe+FE_D3A], [fe+FE_B3], [fe+FE_GWP3], [fe+FE_GWP3], [mdl+MD_C0], [mdl+MD_C0], [cur_r3], MM_TA|MM_TB
    mov rax, [fe+FE_GW3]
    mov [rsp+32], rax
    mov rax, [cur_t2]
    mov [rsp+40], rax
    mov rax, [mdl+MD_F2]
    mov [rsp+48], rax
    mov rcx, [fe+FE_D3B]
    mov rdx, [fe+FE_A2]
    mov r8, [fe+FE_W3]
    mov r9, [fe+FE_D2A]
    call sconv_bwd
    mov r9, [cur_r2]
    imul r9, [mdl+MD_C0]
    mov rcx, [fe+FE_A2]
    mov rdx, [fe+FE_D2A]
    mov r8, rdx
    call relu_bwd
    MM [fe+FE_D2A], [fe+FE_WP2], [fe+FE_D2B], 0, [cur_r2], [mdl+MD_C0], [mdl+MD_C0], [mmbftb]
    MM [fe+FE_D2A], [fe+FE_B2], [fe+FE_GWP2], [fe+FE_GWP2], [mdl+MD_C0], [mdl+MD_C0], [cur_r2], MM_TA|MM_TB
    mov rax, [fe+FE_GW2]
    mov [rsp+32], rax
    mov rax, [cur_t1]
    mov [rsp+40], rax
    mov rax, [mdl+MD_F1]
    mov [rsp+48], rax
    mov rcx, [fe+FE_D2B]
    mov rdx, [fe+FE_A1]
    mov r8, [fe+FE_W2]
    mov r9, [fe+FE_D1]
    call sconv_bwd
    mov r9, [cur_t1]
    imul r9, [mdl+MD_F1]
    imul r9, [cur_b]
    imul r9, [mdl+MD_C0]
    mov rcx, [fe+FE_A1]
    mov rdx, [fe+FE_D1]
    mov r8, rdx
    call relu_bwd
    mov rax, [fe+FE_GW1]
    mov [rsp+32], rax
    mov rax, [cur_tm]
    mov [rsp+40], rax
    mov rax, [mdl+MD_NMEL]
    mov [rsp+48], rax
    mov rcx, [fe+FE_D1]
    mov rdx, [rbx+MB_MEL]
    mov r8, [fe+FE_W1]
    xor r9d, r9d
    call sconv_bwd
    add rsp, 72
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = settings (OP_*, f64), rdx = step number from 1 (for adam's bias correction).
; gradient norm, then adamw on everything. the norm stays on the gpu at d_gn
global model_step
model_step:
    push rbx
    push rsi
    sub rsp, 56
    mov rbx, rcx
    mov rsi, rdx
    mov rcx, [d_gn]
    xor edx, edx
    mov r8d, 1
    CU cuMemsetD32
    KF k_sumsq
    karg 0, [d_grad]
    karg 1, [d_sq]
    karg 2, [mdl+MD_NP]
    mov ecx, NSQ
    mov edx, 1
    mov r8d, 256
    mov r9d, 1
    call launch
    KF k_colsum
    karg 0, [d_sq]
    karg 1, [d_gn]
    karg 2, 1
    karg 3, NSQ
    mov ecx, 1
    call lin
    KF k_adamw
    karg 0, [d_params]
    karg 1, [d_grad]
    karg 2, [d_adm]
    karg 3, [d_adv]
    karg 4, [d_bf]
    karg 5, [d_gn]
    karg 6, [mdl+MD_NP]
    karg 7, [mdl+MD_NDEC]
%macro f32arg 2
    cvtsd2ss xmm0, %2
    movd [kl+KL_ARGS+%1*8], xmm0
%endmacro
    f32arg 8, [rbx+OP_LR]
    f32arg 9, [rbx+OP_B1]
    f32arg 10, [rbx+OP_B2]
    f32arg 11, [rbx+OP_EPS]
    f32arg 12, [rbx+OP_WD]
    f32arg 15, [rbx+OP_CLIP]
    ; bias corrections 1/(1 - b^t)
    movsd xmm0, [rbx+OP_B1]
    call math_log
    cvtsi2sd xmm1, rsi
    mulsd xmm0, xmm1
    call math_exp
    movsd xmm1, [c_one]
    subsd xmm1, xmm0
    movsd xmm0, [c_one]
    divsd xmm0, xmm1
    f32arg 13, xmm0
    movsd xmm0, [rbx+OP_B2]
    call math_log
    cvtsi2sd xmm1, rsi
    mulsd xmm0, xmm1
    call math_exp
    movsd xmm1, [c_one]
    subsd xmm1, xmm0
    movsd xmm0, [c_one]
    divsd xmm0, xmm1
    f32arg 14, xmm0
    mov ecx, 4096
    shl ecx, 8
    call lin
    add rsp, 56
    pop rsi
    pop rbx
    ret

; rcx = device u32 [M]: each encoder frame's most likely token, from the logits of
; the last model_fwd (greedy decoding)
global model_argmax
model_argmax:
    sub rsp, 40
    KF k_argmax
    karg 0, [d_logits]
    karg 1, rcx
    karg 2, [cur_m]
    karg 3, [mdl+MD_V]
    mov rcx, [cur_m]
    call rows8
    add rsp, 40
    ret

; a line about the model: params, flops per frame, memory
global model_summary
model_summary:
    sub rsp, 40
    say "  "
    mov rcx, [mdl+MD_NP]
    call print_dec
    say " params ("
    mov rcx, [mdl+MD_L]
    call print_dec
    say " blocks of "
    mov rcx, [mdl+MD_D]
    call print_dec
    say "), "
    mov rax, [vram]
    shr rax, 20
    mov rcx, rax
    call print_dec
    say " MB on the gpu", 13, 10
    add rsp, 40
    ret
