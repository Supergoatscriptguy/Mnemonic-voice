; tar archives (ustar, with gnu long names), walked in memory. a tar is a run of
; 512 byte headers, each followed by its file padded to 512, and two zero blocks
default rel
bits 64
%include "lib.inc"
%include "data/tar.inc"

section .text

; octal field. rcx = text, edx = max chars. rax = value
octal:
    xor eax, eax
.d:
    test edx, edx
    jz .out
    movzx r8d, byte [rcx]
    sub r8d, '0'
    cmp r8d, 7
    ja .skip                    ; spaces and the trailing 0
    shl rax, 3
    add rax, r8
.skip:
    inc rcx
    dec edx
    jmp .d
.out:
    ret

; rcx = TA, rdx = start, r8 = size
global tar_open
tar_open:
    mov [rcx+TA_POS], rdx
    add rdx, r8
    mov [rcx+TA_END], rdx
    mov qword [rcx+TA_LONG], 0
    ret

; rcx = TA. moves to the next regular file: TA_NAME (zero terminated, in TA_PATH),
; TA_DATA, TA_SIZE. eax = 1, 0 at the end, 2 if the archive is cut off in the
; middle of this file (TA_DATA/TA_SIZE then cover what's there)
global tar_next
tar_next:
    push rbx
    push rsi
    push rdi
    mov rbx, rcx
.header:
    mov rsi, [rbx+TA_POS]
    lea rax, [rsi+512]
    cmp rax, [rbx+TA_END]
    ja .end
    cmp byte [rsi], 0
    je .end                     ; the zero blocks at the end
    lea rcx, [rsi+124]
    mov edx, 12
    call octal
    mov [rbx+TA_SIZE], rax
    lea rdx, [rsi+512]
    mov [rbx+TA_DATA], rdx
    add rax, 511
    and rax, -512
    add rdx, rax
    mov [rbx+TA_POS], rdx
    mov al, [rsi+156]           ; type
    cmp al, 'L'
    je .long
    cmp al, '0'
    je .file
    test al, al
    je .file
    jmp .header                 ; dirs, links, pax headers: skip
.long:
    ; gnu: the next header's name is this file's contents
    mov rdi, [rbx+TA_DATA]
    mov [rbx+TA_LONG], rdi
    jmp .header
.file:
    lea rdi, [rbx+TA_PATH]
    mov rdx, [rbx+TA_LONG]
    test rdx, rdx
    jnz .copy
    ; ustar: prefix/name
    cmp dword [rsi+257], 'usta'
    jne .name
    cmp byte [rsi+345], 0
    je .name
    lea rdx, [rsi+345]
    mov ecx, 155
.pre:
    mov al, [rdx]
    test al, al
    jz .slash
    mov [rdi], al
    inc rdi
    inc rdx
    dec ecx
    jnz .pre
.slash:
    mov byte [rdi], '/'
    inc rdi
.name:
    mov rdx, rsi
    mov ecx, 100
    jmp .take
.copy:
    mov ecx, TA_PATHLEN - 1
    mov qword [rbx+TA_LONG], 0
.take:
    mov al, [rdx]
    test al, al
    jz .named
    mov [rdi], al
    inc rdi
    inc rdx
    dec ecx
    jnz .take
.named:
    mov byte [rdi], 0
    lea rax, [rbx+TA_PATH]
    mov [rbx+TA_NAME], rax
    mov eax, 1
    mov rdx, [rbx+TA_DATA]
    add rdx, [rbx+TA_SIZE]
    cmp rdx, [rbx+TA_END]
    jbe .out
    mov rdx, [rbx+TA_END]       ; cut off: hand back what's there
    sub rdx, [rbx+TA_DATA]
    mov [rbx+TA_SIZE], rdx
    mov rdx, [rbx+TA_END]
    mov [rbx+TA_POS], rdx
    mov eax, 2
    jmp .out
.end:
    xor eax, eax
.out:
    pop rdi
    pop rsi
    pop rbx
    ret
