; stage 4 test: scoring. greedy ctc decoding, words exact and normalized, and the
; word edit distance, on cases worked out by hand
; uses: asr\score
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"

extern ctc_greedy, words, wdist

BLANK   equ 9

section .rdata
frames  dd 5, 5, BLANK, 5, 3, 3, BLANK, BLANK, 7, 7
want    dw 5, 5, 3, 7
; (reference, hypothesis, normalize, edits)
r1      db "the cat sat on the mat"
r1n     equ $ - r1
h1      db "the cat sit on mat"
h1n     equ $ - h1
r2      db "Hello, world."
r2n     equ $ - r2
h2      db "hello world"
h2n     equ $ - h2
r3      db "'Tis a well-known fact", 0xe2, 0x80, 0x94, "isn't it?"
r3n     equ $ - r3
h3      db "tis a well known fact isn't it"
h3n     equ $ - h3
r4      db "a b c"
r4n     equ $ - r4

section .bss
alignb 8
got     resw 16
ha      resq 64
hb      resq 64

section .text

; rcx = ref, edx = len, r8 = hyp, r9d = len, [rsp+40] = normalize. rax = edits
edits:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, r8
    mov edi, r9d
    mov r12d, [rsp+40+72]
    mov r8d, r12d
    lea r9, [ha]
    call words
    mov rbx, rax
    mov rcx, rsi
    mov edx, edi
    mov r8d, r12d
    lea r9, [hb]
    call words
    lea rcx, [ha]
    mov rdx, rbx
    lea r8, [hb]
    mov r9, rax
    call wdist
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

%macro EDITS 6                  ; ref, len, hyp, len, normalize, want
    mov qword [rsp+32], %5
    lea rcx, [%1]
    mov edx, %2
    lea r8, [%3]
    mov r9d, %4
    call edits
    cmp rax, %6
%endmacro

global start
start:
    sub rsp, 56
    call lib_init
    lea rcx, [frames]
    mov edx, 10
    mov r8d, BLANK
    lea r9, [got]
    call ctc_greedy
    cmp rax, 4
    jne .g
    mov rax, [got]
    cmp rax, [want]
.g:
    check e, "greedy ctc: 5 5 _ 5 3 3 _ _ 7 7 -> 5 5 3 7"
    EDITS r1, r1n, h1, h1n, 0, 2
    check e, "the cat sat on the mat / the cat sit on mat: 2 edits"
    EDITS r1, r1n, r1, r1n, 0, 0
    check e, "a transcript against itself: 0"
    EDITS r2, r2n, h2, h2n, 0, 2
    check e, "Hello, world. / hello world, as written: 2"
    EDITS r2, r2n, h2, h2n, 1, 0
    check e, "...normalized: 0"
    EDITS r3, r3n, h3, h3n, 1, 0
    check e, "quotes, hyphen, em dash, question mark normalized away: 0"
    EDITS r4, r4n, r4, 0, 0, 3
    check e, "nothing said for a b c: 3 deletions"
    EDITS r4, 0, r4, r4n, 0, 3
    check e, "a b c said over silence: 3 insertions"
    jmp t_done
