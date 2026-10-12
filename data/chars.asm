; chars DIR...: every unicode character in the transcripts of DIR\*.shard, counted,
; the most common first, with an example of each rare one
; uses: data\shard
default rel
bits 64
%include "lib.inc"
%include "data/shard.inc"

extern ExitProcess

MAXCP   equ 0x110000

section .rdata
s_pat   db "\*.shard", 0
usage   db "usage: chars DIR...", 13, 10, 0

section .bss
alignb 8
counts  resq MAXCP
where   resq MAXCP              ; the first text a character showed up in
wlen    resd MAXCP
sr      resb SR_SIZE
pattern resb 1024
found   resb 1 << 20
order   resd 65536
norder  resq 1
texts   resq 1
bytes   resq 1
bad     resq 1
line    resb 256

section .text

; rcx = text, edx = its length. counts its code points
count:
    push rbx
    push rsi
    push rdi
    mov rsi, rcx
    lea rdi, [rcx+rdx]
    mov rbx, rcx
.c:
    cmp rsi, rdi
    jae .out
    movzx eax, byte [rsi]
    cmp eax, 0x80
    jb .one
    cmp eax, 0xe0
    jb .two
    cmp eax, 0xf0
    jb .three
    ; four bytes
    and eax, 7
    shl eax, 18
    movzx edx, byte [rsi+1]
    and edx, 63
    shl edx, 12
    or eax, edx
    movzx edx, byte [rsi+2]
    and edx, 63
    shl edx, 6
    or eax, edx
    movzx edx, byte [rsi+3]
    and edx, 63
    or eax, edx
    add rsi, 4
    jmp .got
.three:
    and eax, 15
    shl eax, 12
    movzx edx, byte [rsi+1]
    and edx, 63
    shl edx, 6
    or eax, edx
    movzx edx, byte [rsi+2]
    and edx, 63
    or eax, edx
    add rsi, 3
    jmp .got
.two:
    cmp eax, 0xc0
    jb .bad
    and eax, 31
    shl eax, 6
    movzx edx, byte [rsi+1]
    and edx, 63
    or eax, edx
    add rsi, 2
    jmp .got
.bad:
    inc qword [bad]
    inc rsi
    jmp .c
.one:
    inc rsi
.got:
    cmp eax, MAXCP
    jae .c
    lea rdx, [counts]
    cmp qword [rdx+rax*8], 0
    jne .seen
    lea rdx, [where]
    mov [rdx+rax*8], rbx
    lea rdx, [wlen]
    mov ecx, edi
    sub ecx, ebx
    mov [rdx+rax*4], ecx
    lea rdx, [counts]
.seen:
    inc qword [rdx+rax*8]
    jmp .c
.out:
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = code point, rdi = out. utf-8 bytes, rax = end
utf8:
    mov eax, ecx
    cmp eax, 0x80
    jae .two
    mov [rdi], al
    lea rax, [rdi+1]
    ret
.two:
    cmp eax, 0x800
    jae .three
    shr eax, 6
    or al, 0xc0
    mov [rdi], al
    mov eax, ecx
    and al, 63
    or al, 0x80
    mov [rdi+1], al
    lea rax, [rdi+2]
    ret
.three:
    cmp eax, 0x10000
    jae .four
    shr eax, 12
    or al, 0xe0
    mov [rdi], al
    mov eax, ecx
    shr eax, 6
    and al, 63
    or al, 0x80
    mov [rdi+1], al
    mov eax, ecx
    and al, 63
    or al, 0x80
    mov [rdi+2], al
    lea rax, [rdi+3]
    ret
.four:
    shr eax, 18
    or al, 0xf0
    mov [rdi], al
    mov eax, ecx
    shr eax, 12
    and al, 63
    or al, 0x80
    mov [rdi+1], al
    mov eax, ecx
    shr eax, 6
    and al, 63
    or al, 0x80
    mov [rdi+2], al
    mov eax, ecx
    and al, 63
    or al, 0x80
    mov [rdi+3], al
    lea rax, [rdi+4]
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
    cmp qword [argc], 2
    jae .dirs
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
.dirs:
    mov r12d, 1
.dir:
    cmp r12, [argc]
    jae .counted
    lea rax, [argv]
    mov rdx, [rax+r12*8]
    lea rcx, [pattern]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_pat]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [pattern]
    lea rdx, [found]
    mov r8d, 1 << 20
    call file_find
    mov rbx, rax
    mov rsi, rdx
.shard:
    test rbx, rbx
    jz .nextdir
    mov rcx, [rsi]
    lea rdx, [sr]
    call shard_open
    test eax, eax
    jz .skip
    xor r13d, r13d
.utt:
    cmp r13, [sr+SR_COUNT]
    jae .closed
    imul rax, r13, UT_SIZE
    add rax, [sr+SR_INDEX]
    mov rcx, [rax+UT_TEXT]
    add rcx, [sr+SR_BASE]
    mov edx, [rax+UT_TLEN]
    add [bytes], rdx
    inc qword [texts]
    call count
    inc r13
    jmp .utt
.closed:                        ; left mapped: the examples point into it
.skip:
    add rsi, 8
    dec rbx
    jmp .shard
.nextdir:
    inc r12
    jmp .dir

.counted:
    ; every code point seen, most common first (selection, there aren't many)
    xor ecx, ecx
    lea rdx, [counts]
    lea r8, [order]
.collect:
    cmp qword [rdx+rcx*8], 0
    je .nc
    mov rax, [norder]
    cmp rax, 65536
    jae .nc
    mov [r8+rax*4], ecx
    inc qword [norder]
.nc:
    inc ecx
    cmp ecx, MAXCP
    jb .collect
    mov rcx, [texts]
    call print_dec
    say " transcripts, "
    mov rcx, [bytes]
    call print_dec
    say " bytes, "
    mov rcx, [norder]
    call print_dec
    say " different characters, "
    mov rcx, [bad]
    call print_dec
    say " bad utf-8 bytes", 13, 10
    xor r12d, r12d
.sort:
    cmp r12, [norder]
    jae .done
    ; find the biggest from r12 on, swap it to r12
    lea r8, [order]
    lea r9, [counts]
    mov rbx, r12
    mov eax, [r8+r12*4]
    mov r10, [r9+rax*8]
    lea rcx, [r12+1]
.max:
    cmp rcx, [norder]
    jae .swap
    mov eax, [r8+rcx*4]
    cmp [r9+rax*8], r10
    jbe .nm
    mov r10, [r9+rax*8]
    mov rbx, rcx
.nm:
    inc rcx
    jmp .max
.swap:
    mov eax, [r8+r12*4]
    mov edx, [r8+rbx*4]
    mov [r8+r12*4], edx
    mov [r8+rbx*4], eax
    ; print it: U+XXXX  char  count  (an example if it's rare)
    mov r13d, edx
    say "  U+"
    mov ecx, r13d
    mov edx, 4
    call print_hex
    say "  "
    lea rdi, [line]
    cmp r13d, 32
    ja .vis
    mov byte [rdi], '.'
    inc rdi
    jmp .shown
.vis:
    mov ecx, r13d
    call utf8
    mov rdi, rax
.shown:
    lea rcx, [line]
    mov rdx, rdi
    sub rdx, rcx
    call print
    say "  "
    lea rax, [counts]
    mov rcx, [rax+r13*8]
    call print_dec
    lea rax, [counts]
    cmp qword [rax+r13*8], 2000
    jae .eol
    cmp r13d, 128
    jb .eol
    say "   e.g. "
    lea rax, [where]
    mov rcx, [rax+r13*8]
    lea rax, [wlen]
    mov edx, [rax+r13*4]
    cmp edx, 90
    jbe .ex
    mov edx, 90
.ex:
    call print
.eol:
    say 13, 10
    inc r12
    jmp .sort
.done:
    call con_restore
    xor ecx, ecx
    call ExitProcess
