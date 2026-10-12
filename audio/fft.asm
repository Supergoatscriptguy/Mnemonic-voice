; complex fft of any power of two, f32 (re, im) pairs, in place, and real
; convolution through it (room reverb for augmentation). fft_init makes the
; twiddles e^(-2 pi i k / N) for the largest size, in f64 then rounded; smaller
; sizes take every (N/n)-th one
default rel
bits 64
%include "lib.inc"

section .rdata
align 16
negmask dd 0x80000000, 0, 0, 0
align 8
c_m2pi  dq -6.283185307179586
c_q     dq 0.25

section .bss
alignb 8
tw      resq 1                  ; f32 (cos, sin), 2^(tmax-1) of them
tmax    resq 1

section .text

; ecx = log2 of the largest size fft will be asked for
global fft_init
fft_init:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    mov [tmax], rcx
    mov esi, 1
    shl rsi, cl                 ; N
    mov rcx, rsi
    shl rcx, 2                  ; N/2 pairs of 8 bytes
    call mem_alloc
    mov [tw], rax
    mov rdi, rax
    xor ebx, ebx
.k:
    mov rax, rsi
    shr rax, 1
    cmp rbx, rax
    jae .done
    cvtsi2sd xmm0, rbx
    cvtsi2sd xmm1, rsi
    divsd xmm0, xmm1
    mulsd xmm0, [c_m2pi]
    movsd [rsp+32], xmm0
    call math_cos
    cvtsd2ss xmm0, xmm0
    movss [rdi+rbx*8], xmm0
    movsd xmm0, [rsp+32]
    call math_sin
    cvtsd2ss xmm0, xmm0
    movss [rdi+rbx*8+4], xmm0
    inc rbx
    jmp .k
.done:
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = data (n complex f32), edx = log2 n (<= fft_init's), r8d = 1 for the inverse
; (not scaled: divide by n yourself)
global fft
fft:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    mov rsi, rcx
    mov ecx, edx
    mov r12d, 1
    shl r12, cl                 ; n
    mov r14d, r8d
    ; bit reversed order
    xor edi, edi                ; j, i reversed
    xor ebx, ebx                ; i
.br:
    cmp rbx, r12
    jae .brd
    cmp rbx, rdi
    jae .nos
    mov rax, [rsi+rbx*8]
    mov rdx, [rsi+rdi*8]
    mov [rsi+rbx*8], rdx
    mov [rsi+rdi*8], rax
.nos:
    mov rax, r12
    shr rax, 1
.bit:
    test rdi, rax
    jz .set
    xor rdi, rax
    shr rax, 1
    jmp .bit
.set:
    or rdi, rax
    inc rbx
    jmp .br
.brd:
    mov r15, 2                  ; butterfly span
.stage:
    cmp r15, r12
    ja .done
    mov r8, r15
    shr r8, 1                   ; half
    mov ecx, [tmax]
    mov eax, 1
    shl rax, cl
    xor edx, edx
    div r15
    mov r10, rax                ; twiddle stride
    mov r11, [tw]
    xor ebx, ebx
.grp:
    cmp rbx, r12
    jae .stn
    xor ecx, ecx                ; k
    xor edi, edi                ; its twiddle
.kk:
    cmp rcx, r8
    jae .grpn
    movss xmm0, [r11+rdi*8]     ; wr
    movss xmm1, [r11+rdi*8+4]   ; wi
    test r14d, r14d
    jz .fw
    xorps xmm1, [negmask]
.fw:
    lea rax, [rbx+rcx]
    lea rdx, [rax+r8]
    movss xmm2, [rsi+rdx*8]     ; b
    movss xmm3, [rsi+rdx*8+4]
    movaps xmm4, xmm0
    mulss xmm4, xmm2
    movaps xmm5, xmm1
    mulss xmm5, xmm3
    subss xmm4, xmm5            ; (w b).re
    mulss xmm0, xmm3
    mulss xmm1, xmm2
    addss xmm0, xmm1            ; (w b).im
    movss xmm2, [rsi+rax*8]     ; a
    movss xmm3, [rsi+rax*8+4]
    movaps xmm5, xmm2
    addss xmm5, xmm4
    movss [rsi+rax*8], xmm5
    subss xmm2, xmm4
    movss [rsi+rdx*8], xmm2
    movaps xmm5, xmm3
    addss xmm5, xmm0
    movss [rsi+rax*8+4], xmm5
    subss xmm3, xmm0
    movss [rsi+rdx*8+4], xmm3
    inc rcx
    add rdi, r10
    jmp .kk
.grpn:
    add rbx, r15
    jmp .grp
.stn:
    shl r15, 1
    jmp .stage
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; y = x convolved with h, the first n outputs: y[t] = sum_j h[j] x[t-j].
; rcx = x (f32, n), edx = n, r8 = h (f32, m), r9d = m, [rsp+40] = scratch: 2^k
; complex f32 with 2^k >= n + m - 1, [rsp+48] = that k, [rsp+56] = y (f32, n, may
; be x). one fft does both: z = x + i h, then Z[k] and conj Z[-k] give X and H, and
; Y = X H = -i (A^2 - B^2) / 4 with A = Z[k], B = conj Z[-k]
global fft_conv
fft_conv:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rsi, rcx                ; x
    mov r12d, edx               ; n
    mov rdi, r8                 ; h
    mov r13d, r9d               ; m
    mov rbx, [rsp+40+88]        ; scratch
    mov rcx, [rsp+48+88]
    mov r14d, 1
    shl r14, cl                 ; P
    ; z = x + i h, zeros after
    xor eax, eax
.z:
    cmp rax, r14
    jae .zd
    xorps xmm0, xmm0
    xorps xmm1, xmm1
    cmp rax, r12
    jae .zx
    movss xmm0, [rsi+rax*4]
.zx:
    cmp rax, r13
    jae .zh
    movss xmm1, [rdi+rax*4]
.zh:
    movss [rbx+rax*8], xmm0
    movss [rbx+rax*8+4], xmm1
    inc rax
    jmp .z
.zd:
    mov rcx, rbx
    mov edx, [rsp+48+88]
    xor r8d, r8d
    call fft
    ; Y[k] and Y[P-k] from Z[k] and Z[P-k], a pair at a time (k = 0 and P/2 pair
    ; with themselves)
    movss xmm7, [c_q4]
    xor ecx, ecx                ; k
.y:
    mov rax, r14
    shr rax, 1
    cmp rcx, rax
    ja .yd
    mov rdx, r14
    sub rdx, rcx
    lea rax, [r14-1]
    and rdx, rax                ; (P - k) mod P
    ; A = Z[k], B = conj Z[P-k]
    movss xmm0, [rbx+rcx*8]     ; a.re
    movss xmm1, [rbx+rcx*8+4]   ; a.im
    movss xmm2, [rbx+rdx*8]     ; b.re
    movss xmm3, [rbx+rdx*8+4]
    xorps xmm3, [negmask]       ; b.im
    ; Y[k] = -i (A^2 - B^2) / 4.  A^2 = (ar^2 - ai^2, 2 ar ai)
    call .sq                    ; xmm4, xmm5 = A^2 - B^2
    ; -i (u + iv) = v - iu
    movaps xmm6, xmm5
    mulss xmm6, xmm7
    mulss xmm4, xmm7
    xorps xmm4, [negmask]
    ; Y[P-k]: A' = Z[P-k], B' = conj Z[k] (from the same four numbers)
    movss [rsp+32], xmm6
    movss [rsp+36], xmm4
    movss xmm0, [rbx+rdx*8]
    movss xmm1, [rbx+rdx*8+4]
    movss xmm2, [rbx+rcx*8]
    movss xmm3, [rbx+rcx*8+4]
    xorps xmm3, [negmask]
    call .sq
    movaps xmm6, xmm5
    mulss xmm6, xmm7
    mulss xmm4, xmm7
    xorps xmm4, [negmask]
    movss [rbx+rdx*8], xmm6
    movss [rbx+rdx*8+4], xmm4
    movss xmm6, [rsp+32]
    movss xmm4, [rsp+36]
    movss [rbx+rcx*8], xmm6
    movss [rbx+rcx*8+4], xmm4
    inc rcx
    jmp .y
.yd:
    mov rcx, rbx
    mov edx, [rsp+48+88]
    mov r8d, 1
    call fft
    ; y = re / P, the first n
    cvtsi2ss xmm1, r14
    mov eax, __?float32?__(1.0)
    movd xmm0, eax
    divss xmm0, xmm1
    mov rdi, [rsp+56+88]
    xor eax, eax
.o:
    cmp rax, r12
    jae .od
    movss xmm1, [rbx+rax*8]
    mulss xmm1, xmm0
    movss [rdi+rax*4], xmm1
    inc rax
    jmp .o
.od:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
; xmm0, xmm1 = A, xmm2, xmm3 = B -> xmm4, xmm5 = A^2 - B^2 (uses xmm6)
.sq:
    movaps xmm4, xmm0
    mulss xmm4, xmm0
    movaps xmm6, xmm1
    mulss xmm6, xmm1
    subss xmm4, xmm6            ; ar^2 - ai^2
    movaps xmm5, xmm0
    mulss xmm5, xmm1
    addss xmm5, xmm5            ; 2 ar ai
    movaps xmm6, xmm2
    mulss xmm6, xmm2
    subss xmm4, xmm6
    movaps xmm6, xmm3
    mulss xmm6, xmm3
    addss xmm4, xmm6            ; - (br^2 - bi^2)
    movaps xmm6, xmm2
    mulss xmm6, xmm3
    addss xmm6, xmm6
    subss xmm5, xmm6            ; - 2 br bi
    ret

section .rdata
align 4
c_q4    dd 0.25
