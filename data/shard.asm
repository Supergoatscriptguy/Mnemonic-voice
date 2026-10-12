; reading shards (format in shard.inc): map the file, check the trailer
default rel
bits 64
%include "lib.inc"
%include "data/shard.inc"

section .text

; rcx = path, rdx = SR. eax = 1 if it's a shard
global shard_open
shard_open:
    push rbx
    sub rsp, 32
    mov rbx, rdx
    call file_map
    test rax, rax
    jz .no
    mov [rbx+SR_BASE], rax
    mov [rbx+SR_LEN], rdx
    cmp rdx, SH_SIZE
    jb .no
    lea rcx, [rax+rdx-SH_SIZE]
    mov r8, 'MVSHARD1'
    cmp [rcx+SH_MAGIC], r8
    jne .no
    mov r8, [rcx+SH_COUNT]
    mov [rbx+SR_COUNT], r8
    mov r9, [rcx+SH_SAMPLES]
    mov [rbx+SR_SAMPLES], r9
    mov r9, [rcx+SH_INDEX]
    ; the index has to fit between the blobs and the trailer
    imul r10, r8, UT_SIZE
    add r10, r9
    add r10, SH_SIZE
    cmp r10, rdx
    jne .no
    add r9, rax
    mov [rbx+SR_INDEX], r9
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop rbx
    ret

; rcx = SR
global shard_close
shard_close:
    mov rcx, [rcx+SR_BASE]
    jmp file_unmap
