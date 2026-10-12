; the word language model: what makes a word, the n-gram keys, loading a model and
; its stupid backoff score. a word is a run of letters (lower cased), digits,
; apostrophes and non-ascii bytes; anything else ends it. lmbuild writes models,
; the beam search reads them
default rel
bits 64
%include "lib.inc"
%include "asr/lm.inc"

section .rdata
align 8
c_alpha dq -0.916290731874155   ; ln 0.4, each step of backing off
c_unk   dq -0.693147180559945   ; an unseen word counts as half a sighting
; each byte as it goes into a word, 0 = it ends the word
global lm_wtab
lm_wtab:
%assign b 0
%rep 256
  %if b >= 'A' && b <= 'Z'
    db b + 32
  %elif (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == 0x27 || b >= 0x80
    db b
  %else
    db 0
  %endif
  %assign b b + 1
%endrep

section .text

; rax = the hash of an empty word (fnv-1a 64; a byte goes in as h = (h ^ b) * prime)
lm_wstart:
    mov rax, 0xcbf29ce484222325
    ret

; rax = hash so far, ecx = a byte. rax = with it in, or unchanged with edx = 0 if
; the byte ends a word (edx = 1 if it went in)
lm_wbyte:
    lea rdx, [lm_wtab]
    movzx ecx, cl
    movzx ecx, byte [rdx+rcx]
    xor edx, edx
    test ecx, ecx
    jz .out
    xor rax, rcx
    mov rdx, 0x100000001b3
    imul rax, rdx
    mov edx, 1
.out:
    ret

; rax = murmur3's 64-bit finalizer of rax. touches rdx
fmix:
    mov rdx, rax
    shr rdx, 33
    xor rax, rdx
    mov rdx, 0xff51afd7ed558ccd
    imul rax, rdx
    mov rdx, rax
    shr rdx, 33
    xor rax, rdx
    mov rdx, 0xc4ceb9fe1a85ec53
    imul rax, rdx
    mov rdx, rax
    shr rdx, 33
    xor rax, rdx
    ret

; rcx = w1, rdx = w2: rax = the pair's key (never 0). touches rcx, rdx, r8
lm_key2:
    mov r8, rcx
    mov rax, 0x9e3779b97f4a7c15
    add rax, rdx
    call fmix
    xor rax, r8
    call fmix
    jmp nonzero

; rcx = w1, rdx = w2, r8 = w3: rax = the triple's key. touches rcx, rdx, r8, r9
lm_key3:
    mov r9, r8
    call lm_key2
    mov rcx, rax
    mov rax, 0x3c6ef372fe94f82a
    add rax, r9
    call fmix
    xor rax, rcx
    call fmix
nonzero:
    test rax, rax
    jnz .r
    inc rax
.r:
    ret

; rcx = keys, rdx = counts, r8 = slots - 1, r9 = key. eax = its count, 0 if absent.
; touches r10, r11
lm_find:
    mov r10, r9
.p:
    and r10, r8
    mov r11, [rcx+r10*8]
    test r11, r11
    jz .no
    cmp r11, r9
    je .yes
    inc r10
    jmp .p
.yes:
    mov eax, [rdx+r10*4]
    ret
.no:
    xor eax, eax
    ret

; rcx = path, rdx = LM to fill. eax = 1 if it's a model
lm_load:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rdx
    call file_map
    test rax, rax
    jz .no
    mov rsi, rax
    mov rax, LM_MAGIC
    cmp [rsi+LH_MAGIC], rax
    jne .no
    mov [rbx+LM_BASE], rsi
    mov rax, [rsi+LH_K1]
    add rax, rsi
    mov [rbx+LM_K1], rax
    mov rax, [rsi+LH_C1]
    add rax, rsi
    mov [rbx+LM_C1], rax
    mov rax, [rsi+LH_N1]
    dec rax
    mov [rbx+LM_M1], rax
    mov rax, [rsi+LH_K2]
    add rax, rsi
    mov [rbx+LM_K2], rax
    mov rax, [rsi+LH_C2]
    add rax, rsi
    mov [rbx+LM_C2], rax
    mov rax, [rsi+LH_N2]
    dec rax
    mov [rbx+LM_M2], rax
    mov rax, [rsi+LH_K3]
    add rax, rsi
    mov [rbx+LM_K3], rax
    mov rax, [rsi+LH_C3]
    add rax, rsi
    mov [rbx+LM_C3], rax
    mov rax, [rsi+LH_N3]
    dec rax
    mov [rbx+LM_M3], rax
    cvtsi2sd xmm0, qword [rsi+LH_WORDS]
    call math_log
    movsd [rbx+LM_LOGN], xmm0
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = LM, rdx = w1, r8 = w2, r9 = w3. xmm0 = ln S(w3 | w1 w2): the triple's count
; over its context's if it was seen, else 0.4 times the pair's, else 0.4^2 times
; the word's share of all words
lm_score:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
    ; the triple
    mov rcx, rsi
    mov rdx, rdi
    mov r8, r12
    call lm_key3
    mov r9, rax
    mov rcx, [rbx+LM_K3]
    mov rdx, [rbx+LM_C3]
    mov r8, [rbx+LM_M3]
    call lm_find
    test eax, eax
    jz .pair
    mov r13d, eax
    mov rcx, rsi
    mov rdx, rdi
    call lm_key2
    mov r9, rax
    mov rcx, [rbx+LM_K2]
    mov rdx, [rbx+LM_C2]
    mov r8, [rbx+LM_M2]
    call lm_find
    test eax, eax
    jz .pair
    xorpd xmm1, xmm1
    jmp .ratio
.pair:
    movsd xmm1, [c_alpha]
    movsd [rsp+32], xmm1
    mov rcx, rdi
    mov rdx, r12
    call lm_key2
    mov r9, rax
    mov rcx, [rbx+LM_K2]
    mov rdx, [rbx+LM_C2]
    mov r8, [rbx+LM_M2]
    call lm_find
    test eax, eax
    jz .word
    mov r13d, eax
    mov rcx, [rbx+LM_K1]
    mov rdx, [rbx+LM_C1]
    mov r8, [rbx+LM_M1]
    mov r9, rdi
    call lm_find
    test eax, eax
    jz .word
    movsd xmm1, [rsp+32]
.ratio:
    ; xmm1 + ln(r13 / eax)
    movsd [rsp+32], xmm1
    cvtsi2sd xmm0, r13d
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    call math_log
    addsd xmm0, [rsp+32]
    jmp .out
.word:
    movsd xmm1, [c_alpha]
    addsd xmm1, xmm1
    movsd [rsp+32], xmm1
    mov rcx, [rbx+LM_K1]
    mov rdx, [rbx+LM_C1]
    mov r8, [rbx+LM_M1]
    mov r9, r12
    call lm_find
    test eax, eax
    jz .unk
    cvtsi2sd xmm0, eax
    call math_log
    jmp .share
.unk:
    movsd xmm0, [c_unk]
.share:
    subsd xmm0, [rbx+LM_LOGN]
    addsd xmm0, [rsp+32]
.out:
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

global lm_wstart, lm_wbyte, lm_key2, lm_key3, lm_load, lm_score, lm_find
