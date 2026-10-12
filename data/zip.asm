; zip archives, in memory: the central directory at the end lists every entry and
; where its local header is. stored and deflate entries, crc32 checked. no zip64
default rel
bits 64
%include "lib.inc"
%include "data/inflate.inc"
%include "data/zip.inc"

section .text

; rcx = ZP, rdx = the archive, r8 = its size. eax = 1 if it has a central directory
global zip_open
zip_open:
    mov [rcx+ZP_BASE], rdx
    mov [rcx+ZP_LEN], r8
    ; the end record: 22 bytes plus a comment of up to 64 KB, search back for it
    cmp r8, 22
    jb .no
    lea r9, [rdx+r8-22]
    mov r10, rdx
    cmp r8, 22 + 65535
    jbe .scan
    lea r10, [rdx+r8-22-65535]
.scan:
    cmp dword [r9], 0x06054b50
    je .found
    dec r9
    cmp r9, r10
    jae .scan
.no:
    xor eax, eax
    ret
.found:
    movzx eax, word [r9+10]     ; entries
    mov [rcx+ZP_LEFT], rax
    mov eax, [r9+16]            ; central directory offset
    add rax, rdx
    mov [rcx+ZP_NEXT], rax
    mov eax, 1
    ret

; rcx = ZP. the next entry: ZP_PATH (zero terminated), ZP_METHOD, ZP_DATA,
; ZP_CSIZE, ZP_USIZE, ZP_CRC. eax = 1, 0 at the end (or if it's broken)
global zip_next
zip_next:
    push rsi
    push rdi
    cmp qword [rcx+ZP_LEFT], 0
    je .no
    dec qword [rcx+ZP_LEFT]
    mov rsi, [rcx+ZP_NEXT]
    cmp dword [rsi], 0x02014b50
    jne .no
    movzx eax, word [rsi+10]
    mov [rcx+ZP_METHOD], eax
    mov eax, [rsi+16]
    mov [rcx+ZP_CRC], eax
    mov eax, [rsi+20]
    mov [rcx+ZP_CSIZE], rax
    mov eax, [rsi+24]
    mov [rcx+ZP_USIZE], rax
    movzx r8d, word [rsi+28]    ; name
    movzx r9d, word [rsi+30]    ; extra
    movzx r10d, word [rsi+32]   ; comment
    ; the name, zero terminated
    lea rdi, [rcx+ZP_PATH]
    mov edx, r8d
    cmp edx, ZP_PATHLEN - 1
    jbe .fits
    mov edx, ZP_PATHLEN - 1
.fits:
    push rcx
    push rsi
    lea rsi, [rsi+46]
    mov ecx, edx
    rep movsb
    mov byte [rdi], 0
    pop rsi
    pop rcx
    ; the next central entry
    lea rax, [rsi+46+r8]
    add rax, r9
    add rax, r10
    mov [rcx+ZP_NEXT], rax
    ; the data sits after the local header, whose name and extra can differ
    mov eax, [rsi+42]
    add rax, [rcx+ZP_BASE]
    cmp dword [rax], 0x04034b50
    jne .no
    movzx r8d, word [rax+26]
    movzx r9d, word [rax+28]
    lea rax, [rax+30+r8]
    add rax, r9
    mov [rcx+ZP_DATA], rax
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    pop rdi
    pop rsi
    ret

; rcx = ZP (after zip_next), rdx = out (ZP_USIZE bytes and a little more).
; eax = 1 if it came out whole with the right crc32
global zip_extract
zip_extract:
    push rbx
    push rsi
    push rdi
    sub rsp, 32 + IF_SIZE
    mov rbx, rcx
    mov rdi, rdx
    cmp dword [rbx+ZP_METHOD], 0
    je .stored
    cmp dword [rbx+ZP_METHOD], 8
    jne .no
    lea rcx, [rsp+32]
    mov rdx, [rbx+ZP_DATA]
    mov r8, [rbx+ZP_CSIZE]
    mov r9d, 1 << 20
    call inflate_init
    xor esi, esi                ; bytes out
.run:
    lea rcx, [rsp+32]
    call inflate_run
    cmp eax, -1
    je .bad
    mov r8, [rsp+32+IF_N]
    sub r8, [rsp+32+IF_FROM]
    lea rcx, [rsi+r8]
    cmp rcx, [rbx+ZP_USIZE]
    ja .bad
    mov r9d, eax                ; the run's answer, rep movsb leaves r8/r9 alone
    push rsi
    push rdi
    add rdi, rsi
    mov rsi, [rsp+16+32+IF_OUT]
    add rsi, [rsp+16+32+IF_FROM]
    mov rcx, r8
    rep movsb
    pop rdi
    pop rsi
    add rsi, r8
    cmp r9d, 1
    je .run
    lea rcx, [rsp+32]
    call inflate_free
    cmp rsi, [rbx+ZP_USIZE]
    jne .no
    jmp .check
.bad:
    lea rcx, [rsp+32]
    call inflate_free
    jmp .no
.stored:
    mov rsi, [rbx+ZP_DATA]
    mov rcx, [rbx+ZP_CSIZE]
    push rdi
    rep movsb
    pop rdi
.check:
    xor ecx, ecx
    mov rdx, rdi
    mov r8, [rbx+ZP_USIZE]
    call crc32
    cmp eax, [rbx+ZP_CRC]
    jne .no
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32 + IF_SIZE
    pop rdi
    pop rsi
    pop rbx
    ret
