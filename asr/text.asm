; transcripts -> the text the model learns to write. a light cleanup, decided from
; bin\chars over the real transcripts: book markup (_ *) goes, odd apostrophes,
; dashes and quotes become plain ones, whitespace collapses, and an utterance with
; anything that isn't spoken as written (footnote brackets, math, other scripts)
; is left out
default rel
bits 64
%include "lib.inc"

section .text

; rcx = utf-8 text, rdx = length, r8 = out (room for 3x the length).
; rax = the cleaned length, or -1 to leave the utterance out
global clean_text
clean_text:
    push rbx
    push rsi
    push rdi
    push r12
    mov rsi, rcx
    lea rbx, [rcx+rdx]
    mov rdi, r8
    mov r12, r8
    mov r10d, 1                 ; the last thing written was a space (or nothing yet)
.c:
    cmp rsi, rbx
    jae .end
    mov r11, rsi                ; this character's first byte
    movzx eax, byte [rsi]
    cmp eax, 0x80
    jb .ascii
    ; utf-8: 2, 3 or 4 bytes, each continuation 10xxxxxx
    cmp eax, 0xc2
    jb .drop
    cmp eax, 0xe0
    jb .two
    cmp eax, 0xf0
    jb .three
    cmp eax, 0xf4
    ja .drop
    lea rcx, [rsi+4]
    cmp rcx, rbx
    ja .drop
    and eax, 7
    mov ecx, 3
    jmp .cont
.three:
    lea rcx, [rsi+3]
    cmp rcx, rbx
    ja .drop
    and eax, 15
    mov ecx, 2
    jmp .cont
.two:
    lea rcx, [rsi+2]
    cmp rcx, rbx
    ja .drop
    and eax, 31
    mov ecx, 1
.cont:
    inc rsi
.cb:
    movzx edx, byte [rsi]
    mov r9d, edx
    and r9d, 0xc0
    cmp r9d, 0x80
    jne .drop
    shl eax, 6
    and edx, 63
    or eax, edx
    inc rsi
    dec ecx
    jnz .cb
    ; eax = the code point, [r11, rsi) its bytes
    cmp eax, 0xa0
    je .space
    cmp eax, 0x2014             ; em dash, kept
    je .copy
    cmp eax, 0xa3               ; pound sign
    je .copy
    cmp eax, 0xb0               ; degree sign
    je .copy
    cmp eax, 0x2010             ; hyphens, en dash, minus
    je .hyphen
    cmp eax, 0x2011
    je .hyphen
    cmp eax, 0x2013
    je .hyphen
    cmp eax, 0x2212
    je .hyphen
    cmp eax, 0x2018             ; apostrophes and their look-alikes
    je .apos
    cmp eax, 0x2019
    je .apos
    cmp eax, 0x2bc
    je .apos
    cmp eax, 0xb4
    je .apos
    cmp eax, 0x2bd
    je .apos
    cmp eax, 0x2bb
    je .apos
    cmp eax, 0x201c             ; quotes
    je .quote
    cmp eax, 0x201d
    je .quote
    cmp eax, 0xab
    je .quote
    cmp eax, 0xbb
    je .quote
    cmp eax, 0x2026             ; ellipsis
    je .dots
    ; latin letters with accents, but not the multiply and divide signs
    cmp eax, 0xd7
    je .drop
    cmp eax, 0xf7
    je .drop
    cmp eax, 0xc0
    jb .drop
    cmp eax, 0x24f
    jbe .copy
    jmp .drop

.ascii:
    inc rsi
    cmp eax, ' '
    je .space
    cmp eax, 9
    je .space
    cmp eax, 10
    je .space
    cmp eax, 13
    je .space
    cmp eax, '_'                ; book markup
    je .c
    cmp eax, '*'
    je .c
    cmp eax, '`'
    je .apos
    mov edx, eax
    or edx, 0x20
    cmp edx, 'a'
    jb .notletter
    cmp edx, 'z'
    jbe .one
.notletter:
    cmp eax, '0'
    jb .punct
    cmp eax, '9'
    jbe .one
.punct:
    ; . , ? ! ; : ' " ( ) - & $ %
    cmp eax, '.'
    je .one
    cmp eax, ','
    je .one
    cmp eax, '?'
    je .one
    cmp eax, '!'
    je .one
    cmp eax, ';'
    je .one
    cmp eax, ':'
    je .one
    cmp eax, "'"
    je .one
    cmp eax, '"'
    je .one
    cmp eax, '('
    je .one
    cmp eax, ')'
    je .one
    cmp eax, '-'
    je .one
    cmp eax, '&'
    je .one
    cmp eax, '$'
    je .one
    cmp eax, '%'
    je .one
    jmp .drop                   ; [ ] { } < > # ^ = | \ / ~ @ + and controls

.one:
    mov [rdi], al
    inc rdi
    xor r10d, r10d
    jmp .c
.copy:
    mov rcx, rsi
    sub rcx, r11
    push rsi
    mov rsi, r11
    rep movsb
    pop rsi
    xor r10d, r10d
    jmp .c
.hyphen:
    mov eax, '-'
    jmp .one
.apos:
    mov eax, "'"
    jmp .one
.quote:
    mov eax, '"'
    jmp .one
.dots:
    mov word [rdi], '..'
    mov byte [rdi+2], '.'
    add rdi, 3
    xor r10d, r10d
    jmp .c
.space:
    test r10d, r10d
    jnz .c
    mov byte [rdi], ' '
    inc rdi
    mov r10d, 1
    jmp .c
.end:
    cmp rdi, r12
    je .drop                    ; nothing left
    cmp byte [rdi-1], ' '
    jne .done
    dec rdi
.done:
    mov rax, rdi
    sub rax, r12
    jmp .out
.drop:
    mov rax, -1
.out:
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
