; helpers for the gpu kernel tests: launching, comparing against the cpu's f64,
; bf16 rounding. include after lib.inc, test/check.inc and gpu/cuda.inc. the test
; supplies `back`, a host buffer big enough for anything it brings back to compare

section .rdata
align 16
t_abs   dq 0x7fffffffffffffff, 0x7fffffffffffffff
t_nan   dq 0x7ff8000000000000

section .bss
alignb 8
kl      resb KL_SIZE

section .text

; rcx = module, the function goes in [rdx]
%macro getf 3
    mov rcx, %1
    lea rdx, [%2]
    call gpu_func
    mov [%3], rax
%endmacro

; xmm0 = a new difference, rcx = the running max. a nan sticks (maxsd alone would
; drop it again on the next number)
keepmax:
    movsd xmm1, [rcx]
    ucomisd xmm1, xmm1
    jp .done
    ucomisd xmm0, xmm0
    jp .put
    maxsd xmm0, xmm1
.put:
    movsd [rcx], xmm0
.done:
    ret

; rcx = what the gpu gave, edx = 1 if that's bf16 (else f32), r8 = the cpu's f64,
; r9d = count. xmm0 = max |gpu - cpu| / max |cpu|, nan if any were nan
cmprel:
    xorpd xmm2, xmm2
    xorpd xmm3, xmm3
    xor r11d, r11d
    xor eax, eax
.l:
    test edx, edx
    jnz .b
    cvtss2sd xmm0, [rcx+rax*4]
    jmp .c
.b:
    movzx r10d, word [rcx+rax*2]
    shl r10d, 16
    movd xmm0, r10d
    cvtss2sd xmm0, xmm0
.c:
    movsd xmm1, [r8+rax*8]
    subsd xmm0, xmm1
    ucomisd xmm0, xmm0
    jnp .n
    inc r11d
.n:
    andpd xmm0, [t_abs]
    andpd xmm1, [t_abs]
    maxsd xmm2, xmm0
    maxsd xmm3, xmm1
    inc eax
    cmp eax, r9d
    jb .l
    divsd xmm2, xmm3
    movapd xmm0, xmm2
    test r11d, r11d
    jz .done
    movsd xmm0, [t_nan]
.done:
    ret

; rcx = device buffer, rdx = the cpu's f64, r8d = count, r9d = 1 if bf16.
; brings it back into `back` and compares: xmm0 as cmprel
against:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, rdx
    mov esi, r8d
    mov edi, r9d
    mov rdx, rcx
    lea rcx, [back]
    mov eax, 4
    sub eax, edi
    sub eax, edi
    imul r8d, eax
    call gpu_down
    lea rcx, [back]
    mov edx, edi
    mov r8, rbx
    mov r9d, esi
    call cmprel
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; r8d = f32 bits -> r8w = bf16 bits, round to nearest even (no nans in here)
bf16:
    mov r10d, r8d
    shr r10d, 16
    and r10d, 1
    add r8d, 0x7fff
    add r8d, r10d
    shr r8d, 16
    ret

; rbx = kl with the args in. rax = the function, ecx = how many threads, 256 a block
go_flat:
    sub rsp, 40
    mov [rbx+KL_FUNC], rax
    lea eax, [rcx+255]
    shr eax, 8
    mov dword [rbx+KL_GX], eax
    mov dword [rbx+KL_GY], 1
    mov dword [rbx+KL_GZ], 1
    mov dword [rbx+KL_BX], 256
    mov dword [rbx+KL_BY], 1
    mov dword [rbx+KL_BZ], 1
    mov rcx, rbx
    call gpu_launch
    add rsp, 40
    ret

; rcx = device buffer, edx = bytes. zeroes it
zero:
    sub rsp, 40
    mov r8d, edx
    shr r8d, 2
    xor edx, edx
    CU cuMemsetD32
    add rsp, 40
    ret

; ecx = bytes. rax = device buffer, zeroed
galloc:
    push rbx
    push rsi
    sub rsp, 40
    mov esi, ecx
    call gpu_alloc
    mov rbx, rax
    mov rcx, rax
    mov edx, esi
    call zero
    mov rax, rbx
    add rsp, 40
    pop rsi
    pop rbx
    ret
