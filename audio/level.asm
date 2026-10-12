; how loud the speech in a clip is, apart from its pauses
default rel
bits 64
%include "lib.inc"

section .rdata
align 8
c_10log  dq 4.342944819032518   ; 10 / ln 10
c_tiny   dq 1e-12               ; -120 dB
c_120    dq 120.0
c_half   dq 0.5

section .bss
alignb 4
hist     resd 256

section .text

; rcx = f32 mono samples, rdx = how many, r8d = sample rate, r9d = percentile (90 is
; the speech, 10 the noise floor). the rms of every 20 ms frame, in half-dB bins,
; and that percentile of them. xmm0 = dBFS (f64), -120 under a frame. not thread safe
global level_db
level_db:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    mov rsi, rcx
    mov rdi, rdx
    mov r12d, r9d
    mov eax, r8d
    xor edx, edx
    mov ecx, 50
    div ecx
    mov ebx, eax                ; frame
    lea rcx, [hist]
    xor eax, eax
.clr:
    mov dword [rcx+rax*4], 0
    inc eax
    cmp eax, 256
    jb .clr
    xor r13d, r13d              ; frames
.fr:
    cmp rdi, rbx
    jb .count
    xorpd xmm1, xmm1
    xor eax, eax
.s:
    cvtss2sd xmm0, [rsi+rax*4]
    mulsd xmm0, xmm0
    addsd xmm1, xmm0
    inc eax
    cmp eax, ebx
    jb .s
    cvtsi2sd xmm0, ebx
    divsd xmm1, xmm0
    maxsd xmm1, [c_tiny]
    movapd xmm0, xmm1
    call math_log
    mulsd xmm0, [c_10log]
    addsd xmm0, [c_120]
    addsd xmm0, xmm0
    cvttsd2si eax, xmm0
    test eax, eax
    jns .lo
    xor eax, eax
.lo:
    cmp eax, 255
    jbe .hi
    mov eax, 255
.hi:
    lea rcx, [hist]
    inc dword [rcx+rax*4]
    inc r13
    lea rsi, [rsi+rbx*4]
    sub rdi, rbx
    jmp .fr
.count:
    test r13, r13
    jz .none
    ; skip the loudest (100 - pct)%, the next bin down is it
    mov eax, 100
    sub eax, r12d
    imul rax, r13
    xor edx, edx
    mov ecx, 100
    div rcx
    lea rcx, [hist]
    mov edx, 256
    xor r8d, r8d
.w:
    dec edx
    add r8d, [rcx+rdx*4]
    cmp r8, rax
    ja .found
    test edx, edx
    jnz .w
.found:
    cvtsi2sd xmm0, edx
    mulsd xmm0, [c_half]
    subsd xmm0, [c_120]
    jmp .out
.none:
    movsd xmm0, [c_120]
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    movapd xmm0, xmm1
.out:
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
