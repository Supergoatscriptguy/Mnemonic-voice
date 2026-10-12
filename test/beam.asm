; stage 4 test: the ctc prefix beam search against brute force. 4 frames of random
; probabilities over two tokens and the blank: every one of the 81 alignments,
; collapsed, gives each output its exact probability. a beam wide enough to keep
; every prefix (no language model) has to find the likeliest output and its log
; probability
; uses: asr\beam asr\lm tokenizer\pretok tokenizer\bpe tokenizer\tok
default rel
bits 64
%include "lib.inc"
%include "tokenizer/tok.inc"
%include "test/check.inc"

extern beam_decode, bm_width, bm_topk, bm_alpha, bm_beta, bm_lm, bm_score

T       equ 4
TRIALS  equ 300

section .rdata
align 8
c_tiny  dq 1e-9
c_neg   dd -1.0e9
s_tok   db "datasets\sptok.bin", 0
n_seq   db "the beam's best output is brute force's, 300 random cases", 0
n_lp    db "and its log probability, worst difference", 0
e_tok   db "can't load the tokenizer", 0

section .bss
alignb 8
rng     resb RNG_SIZE
rows    resd T * VOCAB
lp3     resq T * 3              ; f64 log probabilities of token 0, 1, blank
prob    resq 32                 ; exact probability of each output (index below)
got     resw 64
worst   resq 1
nbad    resq 1

section .text

; an output (length n, tokens as bits) -> its slot: (1 << n) - 1 + bits
; brute force: every alignment's probability added to its output's
brute:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    xor eax, eax
.z:
    lea rcx, [prob]
    mov qword [rcx+rax*8], 0
    inc eax
    cmp eax, 32
    jb .z
    xor ebx, ebx                ; alignment, base 3, frame 0 the lowest digit
.a:
    cmp ebx, 81
    jae .done
    xorpd xmm0, xmm0            ; its log probability
    xor esi, esi                ; output bits
    xor edi, edi                ; output length
    mov r8d, -1                 ; last symbol
    mov eax, ebx
    xor ecx, ecx                ; frame
.f:
    xor edx, edx
    mov r9d, 3
    div r9d                     ; edx = this frame's symbol (2 = blank)
    imul r10d, ecx, 3
    add r10d, edx
    lea r11, [lp3]
    addsd xmm0, [r11+r10*8]
    cmp edx, 2
    je .blank
    cmp edx, r8d
    je .same
    ; a new token
    shl esi, 1
    or esi, edx
    inc edi
.same:
    mov r8d, edx
    jmp .fn
.blank:
    mov r8d, -1
.fn:
    inc ecx
    cmp ecx, T
    jb .f
    ; the first token is the highest bit; the beam's output gets its slot the same way
    call math_exp
    mov eax, 1
    mov ecx, edi
    shl eax, cl
    dec eax
    add eax, esi
    lea rcx, [prob]
    addsd xmm0, [rcx+rax*8]
    movsd [rcx+rax*8], xmm0
    inc ebx
    jmp .a
.done:
    add rsp, 32
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
    sub rsp, 40
    call lib_init
    lea rcx, [s_tok]
    call tok_load
    test eax, eax
    jnz .tok
    lea rcx, [e_tok]
    call fatal
.tok:
    lea rcx, [rng]
    mov edx, 7
    call rng_seed
    mov qword [bm_width], 32
    mov qword [bm_topk], 2
    mov qword [bm_alpha], 0
    mov qword [bm_beta], 0
    mov qword [bm_lm], 0
    ; every row impossible but for token 0, token 1 and the blank
    mov eax, [c_neg]
    xor ecx, ecx
.fill:
    lea rdx, [rows]
    mov [rdx+rcx*4], eax
    inc ecx
    cmp ecx, T * VOCAB
    jb .fill
    xor r12d, r12d
.trial:
    cmp r12d, TRIALS
    jae .report
    ; random probabilities, a frame at a time
    xor ebx, ebx
.fr:
    lea rcx, [rng]
    call rng_float
    mulsd xmm0, [c_09]
    addsd xmm0, [c_005]
    movsd [rsp+32], xmm0
    lea rcx, [rng]
    call rng_float
    mulsd xmm0, [c_09]
    addsd xmm0, [c_005]
    movapd xmm1, xmm0
    movsd xmm0, [rsp+32]
    ; p0 = a, p1 = (1 - a) b, blank = (1 - a)(1 - b), with a, b in [0.05, 0.95]: no
    ; token's so unlikely the beam would prune it
    movsd xmm2, [c_one]
    subsd xmm2, xmm0            ; 1 - a
    movapd xmm3, xmm2
    mulsd xmm3, xmm1            ; p1
    movsd xmm4, [c_one]
    subsd xmm4, xmm1
    mulsd xmm4, xmm2            ; blank
    maxsd xmm0, [c_tiny]
    maxsd xmm3, [c_tiny]
    maxsd xmm4, [c_tiny]
    imul esi, ebx, 3
    lea r8, [lp3]
    movsd [r8+rsi*8], xmm0
    movsd [r8+rsi*8+8], xmm3
    movsd [r8+rsi*8+16], xmm4
    xor edi, edi
.lg:
    lea r8, [lp3]
    movsd xmm0, [r8+rsi*8]
    call math_log
    lea r8, [lp3]
    movsd [r8+rsi*8], xmm0
    cvtsd2ss xmm0, xmm0
    imul eax, ebx, VOCAB
    cmp edi, 2
    jne .tk
    add eax, TOK_BLANK
    jmp .st
.tk:
    add eax, edi
.st:
    lea rcx, [rows]
    movss [rcx+rax*4], xmm0
    inc esi
    inc edi
    cmp edi, 3
    jb .lg
    inc ebx
    cmp ebx, T
    jb .fr
    call brute
    ; the likeliest output by brute force
    xor ebx, ebx
    xor ecx, ecx
    movsd xmm1, [prob]
.mx:
    inc ecx
    cmp ecx, 31
    jae .mxd
    lea rax, [prob]
    movsd xmm0, [rax+rcx*8]
    comisd xmm0, xmm1
    jbe .mx
    movapd xmm1, xmm0
    mov ebx, ecx
    jmp .mx
.mxd:
    movsd [rsp+32], xmm1
    ; the beam's
    lea rcx, [rows]
    mov edx, T
    lea r8, [got]
    call beam_decode
    ; its slot
    mov ecx, eax
    xor esi, esi
    xor edx, edx
.sl:
    cmp edx, ecx
    jae .sld
    lea r8, [got]
    movzx r8d, word [r8+rdx*2]
    shl esi, 1
    or esi, r8d
    inc edx
    jmp .sl
.sld:
    mov eax, 1
    shl eax, cl
    dec eax
    add eax, esi
    cmp eax, ebx
    je .same
    inc qword [nbad]
.same:
    movsd xmm0, [rsp+32]
    call math_log
    subsd xmm0, [bm_score]
    andpd xmm0, [absmask]
    maxsd xmm0, [worst]
    movsd [worst], xmm0
    inc r12d
    jmp .trial
.report:
    xor ecx, ecx
    cmp qword [nbad], 0
    sete cl
    lea rdx, [n_seq]
    call t_ok
    movsd xmm2, [worst]
    xor ecx, ecx
    comisd xmm2, [c_tol]
    setb cl
    lea rdx, [n_lp]
    call t_okf
    jmp t_done

section .rdata
align 16
absmask dq 0x7fffffffffffffff, 0x7fffffffffffffff
c_one   dq 1.0
c_tol   dq 1e-5
c_09    dq 0.9
c_005   dq 0.05
