; flac. for now the stream header: STREAMINFO and which metadata blocks are there
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

section .text

; rcx = bytes, rdx = size, r8 = FI to fill. eax = 1 if it starts with a flac
; stream header. everything in it is big endian
global flac_info
flac_info:
    push rbx
    push rsi
    push rdi
    mov rsi, rcx
    lea rdi, [rcx+rdx]
    mov rbx, r8
    xor eax, eax
    mov [rbx+FI_BLOCKS], eax
    mov [rbx+FI_FRAMES], rax
    cmp rdx, 42
    jb .no
    cmp dword [rsi], 'fLaC'
    jne .no
    add rsi, 4
.block:
    lea rax, [rsi+4]
    cmp rax, rdi
    ja .no
    mov eax, [rsi]
    bswap eax
    mov ecx, eax
    shr ecx, 24
    and ecx, 127                ; type
    mov edx, 1
    shl edx, cl
    or [rbx+FI_BLOCKS], edx
    and eax, 0xffffff           ; length
    add rsi, 4
    test ecx, ecx
    jnz .next
    cmp eax, 34
    jb .no
    ; STREAMINFO
    movzx edx, word [rsi]
    xchg dl, dh
    mov [rbx+FI_MINBS], edx
    movzx edx, word [rsi+2]
    xchg dl, dh
    mov [rbx+FI_MAXBS], edx
    mov edx, [rsi+3]            ; min frame size is bytes 4..6
    bswap edx
    and edx, 0xffffff
    mov [rbx+FI_MINFR], edx
    mov edx, [rsi+6]
    bswap edx
    and edx, 0xffffff
    mov [rbx+FI_MAXFR], edx
    ; bytes 10..17: rate 20 bits, channels-1 3, bits-1 5, total samples 36
    mov rdx, [rsi+10]
    bswap rdx
    mov rcx, rdx
    shr rcx, 44
    mov [rbx+FI_RATE], ecx
    mov rcx, rdx
    shr rcx, 41
    and ecx, 7
    inc ecx
    mov [rbx+FI_CH], ecx
    mov rcx, rdx
    shr rcx, 36
    and ecx, 31
    inc ecx
    mov [rbx+FI_BITS], ecx
    mov rcx, 0xfffffffff
    and rdx, rcx
    mov [rbx+FI_SAMPLES], rdx
    mov rdx, [rsi+18]
    mov [rbx+FI_MD5], rdx
    mov rdx, [rsi+26]
    mov [rbx+FI_MD5+8], rdx
.next:
    test byte [rsi-4], 0x80     ; last block
    lea rsi, [rsi+rax]
    jz .block
    cmp rsi, rdi
    ja .no
    mov [rbx+FI_FRAMES], rsi
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    pop rdi
    pop rsi
    pop rbx
    ret
