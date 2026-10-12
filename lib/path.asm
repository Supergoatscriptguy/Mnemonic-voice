; directory paths
default rel
bits 64
%include "lib.inc"

section .text

; rcx = a directory path. makes it and every directory on the way to it.
; the path is cut up and put back in place, so it has to be writable
global make_path
make_path:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rcx
    lea rsi, [rcx+1]
.scan:
    mov al, [rsi]
    test al, al
    jz .last
    cmp al, '\'
    je .cut
    cmp al, '/'
    jne .next
.cut:
    mov byte [rsi], 0
    mov [rsp+32], al
    mov rcx, rbx
    call make_dir
    mov al, [rsp+32]
    mov [rsi], al
.next:
    inc rsi
    jmp .scan
.last:
    mov rcx, rbx
    call make_dir
    add rsp, 40
    pop rsi
    pop rbx
    ret
