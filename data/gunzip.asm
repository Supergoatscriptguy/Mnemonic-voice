; gunzip in.gz [out]: inflates to out (default: in without .gz, .tgz -> .tar),
; streaming through the window, and checks the trailer's crc32 and length
; uses: data\inflate
default rel
bits 64
%include "lib.inc"
%include "data/inflate.inc"

extern ExitProcess

section .rdata
usage   db "usage: gunzip in.gz [out]", 13, 10, 0
e_open  db "can't read that", 0
e_gz    db "not gzip", 0
e_bad   db "the deflate stream is broken", 0
e_out   db "can't write the output", 0
s_tar   db ".tar", 0
s_tmp   db ".tmp", 0
align 8
c_mb    dq 1e-6

section .bss
alignb 8
inf     resb IF_SIZE
outname resb 1024
tmpname resb 1024

section .text
global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init
    cmp qword [argc], 2
    jae .go
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
.go:
    lea rax, [argv]
    mov rsi, [rax+8]
    ; the output's name
    lea rcx, [outname]
    cmp qword [argc], 3
    jb .derive
    mov rdx, [rax+16]
    call fmt_str
    mov byte [rax], 0
    jmp .named
.derive:
    mov rdx, rsi
    call fmt_str
    mov byte [rax], 0
    cmp dword [rax-4], '.tgz'
    jne .gz
    lea rcx, [rax-4]
    lea rdx, [s_tar]
    call fmt_str
    mov byte [rax], 0
    jmp .named
.gz:
    cmp word [rax-2], 'gz'
    jne .named
    cmp byte [rax-3], '.'
    jne .named
    mov byte [rax-3], 0
.named:
    lea rcx, [tmpname]
    lea rdx, [outname]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_tmp]
    call fmt_str
    mov byte [rax], 0
    mov rcx, rsi
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_open]
    call fatal
.mapped:
    lea rcx, [inf]
    mov r8, rdx
    mov rdx, rax
    mov r9d, 16 << 20
    call gz_open
    test eax, eax
    jnz .opened
    lea rcx, [e_gz]
    call fatal
.opened:
    lea rcx, [tmpname]
    call file_create
    cmp rax, -1
    jne .created
    lea rcx, [e_out]
    call fatal
.created:
    mov rbx, rax
    call time_now
    mov [rsp+32], rax
.run:
    lea rcx, [inf]
    call inflate_run
    mov edi, eax
    cmp edi, -1
    jne .write
    lea rcx, [e_bad]
    call fatal
.write:
    mov rcx, rbx
    mov rdx, [inf+IF_OUT]
    add rdx, [inf+IF_FROM]
    mov r8, [inf+IF_N]
    sub r8, [inf+IF_FROM]
    call file_write
    test eax, eax
    jnz .written
    lea rcx, [e_out]
    call fatal
.written:
    cmp edi, 1
    je .run
    lea rcx, [inf]
    call gz_check
    test eax, eax
    jnz .good
    say "crc32 or length doesn't match the trailer, keeping nothing", 13, 10
    mov rcx, rbx
    call file_close
    lea rcx, [tmpname]
    call file_delete
    mov ecx, 1
    call ExitProcess
.good:
    mov rcx, rbx
    call file_flush
    mov rcx, rbx
    call file_close
    lea rcx, [tmpname]
    lea rdx, [outname]
    call file_replace
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    lea rcx, [outname]
    call print_z
    say ": "
    mov rcx, [inf+IF_TOTAL]
    add rcx, [inf+IF_N]
    mov rbx, rcx
    call print_dec
    say " bytes, crc ok, "
    cvtsi2sd xmm0, rbx
    divsd xmm0, [rsp+40]
    mulsd xmm0, [c_mb]
    xor edx, edx
    call print_fixed
    say " MB/s", 13, 10
    call con_restore
    xor ecx, ecx
    call ExitProcess
