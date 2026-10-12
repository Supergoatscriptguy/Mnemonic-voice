; scoring transcripts: greedy ctc decoding, a transcript's words, and the word
; edit distance that word error rate counts. words compare by a 64-bit hash (fnv-1a)
; of their bytes, exactly as written or normalized:
;   lower case (a-z), . , ? ! ; : " ( ) dropped, hyphens and dashes split words,
;   apostrophes kept inside a word but not at its ends ('quoted' -> quoted)
default rel
bits 64
%include "lib.inc"

MAXW    equ 16384               ; words a hypothesis can have
WMAX    equ 256                 ; bytes a word can have (the rest is ignored)
FNVB    equ 0xcbf29ce484222325
FNVP    equ 0x100000001b3

section .bss
alignb 8
row0    resd MAXW + 1
row1    resd MAXW + 1
wbuf    resb WMAX

section .text

; rcx = most likely token per frame (u32), edx = frames, r8d = the blank, r9 = out
; (u16). runs of the same token count once, blanks go. rax = tokens
global ctc_greedy
ctc_greedy:
    push rbx
    xor eax, eax
    mov r10d, r8d               ; the previous frame's token: a blank to start with
    xor r11d, r11d
.t:
    cmp r11d, edx
    jae .done
    mov ebx, [rcx+r11*4]
    cmp ebx, r10d
    je .same
    mov r10d, ebx
    cmp ebx, r8d
    je .same
    mov [r9+rax*2], bx
    inc eax
.same:
    inc r11d
    jmp .t
.done:
    pop rbx
    ret

; wbuf has r10 bytes of a word: its hash into [r13+rbx*8] (rbx counts), unless
; normalizing leaves it empty. r10 back to 0
flush:
    test r10, r10
    jz .none
    xor ecx, ecx                ; first byte
    mov rdx, r10                ; one past the last
    test r12d, r12d
    jz .hash
    lea r11, [wbuf]
.lead:
    cmp rcx, rdx
    jae .empty
    cmp byte [r11+rcx], "'"
    jne .trail
    inc rcx
    jmp .lead
.trail:
    cmp byte [r11+rdx-1], "'"
    jne .hash
    dec rdx
    jmp .trail
.hash:
    mov rax, FNVB
    mov r11, FNVP
    lea r8, [wbuf]
.h:
    cmp rcx, rdx
    jae .put
    movzx r9d, byte [r8+rcx]
    xor rax, r9
    imul rax, r11
    inc rcx
    jmp .h
.put:
    mov [r13+rbx*8], rax
    inc rbx
.empty:
    xor r10d, r10d
.none:
    ret

; rcx = utf-8 text, rdx = its length, r8d = 1 to normalize, r9 = out: a hash per
; word. rax = words
global words
words:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    mov rsi, rcx
    lea rdi, [rcx+rdx]
    mov r12d, r8d
    mov r13, r9
    xor ebx, ebx
    xor r10d, r10d
.c:
    cmp rsi, rdi
    jae .end
    movzx eax, byte [rsi]
    inc rsi
    cmp al, ' '
    je .sep
    test r12d, r12d
    jz .keep
    ; normalizing
    cmp al, 0xe2                ; an em or en dash, U+2014/2013: E2 80 94/93
    jne .ascii
    lea rcx, [rsi+1]
    cmp rcx, rdi
    jae .keep
    cmp byte [rsi], 0x80
    jne .keep
    mov cl, [rsi+1]
    cmp cl, 0x94
    je .dash
    cmp cl, 0x93
    jne .keep
.dash:
    add rsi, 2
    jmp .sep
.ascii:
    cmp al, '-'
    je .sep
    lea rcx, [drop]
.d:
    mov dl, [rcx]
    test dl, dl
    jz .case
    cmp al, dl
    je .c                       ; punctuation: just gone
    inc rcx
    jmp .d
.case:
    cmp al, 'A'
    jb .keep
    cmp al, 'Z'
    ja .keep
    add al, 32
.keep:
    cmp r10, WMAX
    jae .c
    lea rcx, [wbuf]
    mov [rcx+r10], al
    inc r10
    jmp .c
.sep:
    call flush
    jmp .c
.end:
    call flush
    mov rax, rbx
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
drop    db '.,?!;:"()', 0
section .text

; rcx = reference word hashes, rdx = how many, r8 = hypothesis word hashes, r9 = how
; many (MAXW at most). rax = the fewest substitutions, deletions and insertions
; that turn one into the other
global wdist
wdist:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    mov rsi, rcx
    mov r12, rdx
    mov rdi, r8
    mov r13, r9
    cmp r13, MAXW
    jbe .n
    mov r13, MAXW
.n:
    lea r10, [row0]             ; the previous row
    lea r11, [row1]
    xor eax, eax
.r0:
    mov [r10+rax*4], eax        ; turning nothing into j words: j insertions
    inc rax
    cmp rax, r13
    jbe .r0
    xor ebx, ebx                ; reference words done
.i:
    cmp rbx, r12
    jae .done
    lea eax, [rbx+1]
    mov [r11], eax
    mov r14, [rsi+rbx*8]
    mov ecx, 1                  ; j
.j:
    cmp rcx, r13
    ja .row
    xor edx, edx
    cmp r14, [rdi+rcx*8-8]
    setne dl
    add edx, [r10+rcx*4-4]      ; substitute (or match)
    mov eax, [r10+rcx*4]
    inc eax                     ; delete
    cmp eax, edx
    cmovb edx, eax
    mov eax, [r11+rcx*4-4]
    inc eax                     ; insert
    cmp eax, edx
    cmovb edx, eax
    mov [r11+rcx*4], edx
    inc rcx
    jmp .j
.row:
    xchg r10, r11
    inc rbx
    jmp .i
.done:
    mov eax, [r10+r13*4]
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
