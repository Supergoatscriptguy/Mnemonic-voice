; log-mel features, the same code for training and for the apps.
; 16 kHz, 25 ms periodic hann window, 10 ms hop, 512 point fft, power, 80 slaney
; mel filters (0-8 kHz, area normalized), ln(x + 1e-6). frame t covers samples
; [160t, 160t + 400), zeros past the end, ceil(n / 160) frames in all.
; the 512 point real fft is a 256 point complex one plus the even/odd split
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

section .rdata
align 8
c_2pi    dq 6.283185307179586
c_half   dq 0.5
c_one    dq 1.0
c_zero   dq 0.0
c_binhz  dq 31.25               ; 16000 / 512
c_mlin   dq 66.66666666666667   ; slaney: below 1 kHz, 200/3 Hz per mel
c_mbrk   dq 15.0                ; the mel at 1 kHz
c_logstp dq 0.06875177742094912 ; ln(6.4) / 27
c_khz    dq 1000.0
c_fmax   dq 8000.0
align 4
c_eps    dd 1e-6
c_halff  dd 0.5

section .text

; xmm0 = hz. xmm0 = slaney mel
global hz2mel
hz2mel:
    sub rsp, 40
    comisd xmm0, [c_khz]
    jae .log
    divsd xmm0, [c_mlin]
    add rsp, 40
    ret
.log:
    divsd xmm0, [c_khz]
    call math_log
    divsd xmm0, [c_logstp]
    addsd xmm0, [c_mbrk]
    add rsp, 40
    ret

; xmm0 = slaney mel. xmm0 = hz
global mel2hz
mel2hz:
    sub rsp, 40
    comisd xmm0, [c_mbrk]
    jae .exp
    mulsd xmm0, [c_mlin]
    add rsp, 40
    ret
.exp:
    subsd xmm0, [c_mbrk]
    mulsd xmm0, [c_logstp]
    call math_exp
    mulsd xmm0, [c_khz]
    add rsp, 40
    ret

; rcx = MEL to fill: window, fft twiddles, bit reversal, filterbank
global mel_init
mel_init:
    push rbx
    push rsi
    push rdi
    sub rsp, 96
    ; locals: the edge frequencies' buffer +32, an angle +40, mel(8 kHz) +48
    movdqu [rsp+64], xmm6
    movdqu [rsp+80], xmm7
    mov rbx, rcx
    ; periodic hann: 0.5 - 0.5 cos(2 pi n / 400)
    xor esi, esi
.win:
    cvtsi2sd xmm0, esi
    mulsd xmm0, [c_2pi]
    mov eax, MEL_WINLEN
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    call math_cos
    mulsd xmm0, [c_half]
    movsd xmm1, [c_half]
    subsd xmm1, xmm0
    cvtsd2ss xmm1, xmm1
    movss [rbx+MEL_WIN+rsi*4], xmm1
    inc esi
    cmp esi, MEL_WINLEN
    jb .win
    ; 256 point fft twiddles e^(-2 pi i j / 256), j < 128
    xor esi, esi
.tw:
    cvtsi2sd xmm0, esi
    mulsd xmm0, [c_2pi]
    mov eax, 256
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+40], xmm0
    call math_cos
    cvtsd2ss xmm0, xmm0
    movss [rbx+MEL_TWC+rsi*4], xmm0
    movsd xmm0, [rsp+40]
    call math_sin
    cvtsd2ss xmm0, xmm0
    movss [rbx+MEL_TWS+rsi*4], xmm0
    inc esi
    cmp esi, 128
    jb .tw
    ; the split's e^(-2 pi i k / 512), k <= 256
    xor esi, esi
.rt:
    cvtsi2sd xmm0, esi
    mulsd xmm0, [c_2pi]
    mov eax, 512
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    movsd [rsp+40], xmm0
    call math_cos
    cvtsd2ss xmm0, xmm0
    movss [rbx+MEL_RC+rsi*4], xmm0
    movsd xmm0, [rsp+40]
    call math_sin
    cvtsd2ss xmm0, xmm0
    movss [rbx+MEL_RS+rsi*4], xmm0
    inc esi
    cmp esi, 257
    jb .rt
    ; 8 bit reversal
    xor esi, esi
.rev:
    mov eax, esi
    xor edx, edx
    mov ecx, 8
.rb:
    add edx, edx
    mov r8d, eax
    and r8d, 1
    or edx, r8d
    shr eax, 1
    dec ecx
    jnz .rb
    mov [rbx+MEL_REV+rsi*2], dx
    inc esi
    cmp esi, 256
    jb .rev

    ; filter edges: 82 frequencies evenly spaced in mel from 0 to 8 kHz
    mov ecx, 82 * 8
    call mem_alloc
    mov [rsp+32], rax
    movsd xmm0, [c_fmax]
    call hz2mel
    movsd [rsp+48], xmm0        ; mel of 8 kHz
    xor esi, esi
.edge:
    cvtsi2sd xmm0, esi
    mulsd xmm0, [rsp+48]
    mov eax, MEL_N + 1
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    call mel2hz
    mov rax, [rsp+32]
    movsd [rax+rsi*8], xmm0
    inc esi
    cmp esi, MEL_N + 2
    jb .edge
    ; each filter: a triangle over the fft bins between its edges, scaled by
    ; 2 / (upper - lower). only the bins it touches get stored
    xor esi, esi                ; filter
    xor edi, edi                ; weights so far
.filter:
    mov rax, [rsp+32]
    movsd xmm2, [rax+rsi*8]     ; lower
    movsd xmm3, [rax+rsi*8+8]   ; center
    movsd xmm4, [rax+rsi*8+16]  ; upper
    movapd xmm5, xmm4
    subsd xmm5, xmm2
    movsd xmm6, [c_one]
    addsd xmm6, xmm6
    divsd xmm6, xmm5            ; the slaney scale
    mov word [rbx+MEL_FB0+rsi*2], -1
    mov word [rbx+MEL_FBN+rsi*2], 0
    mov [rbx+MEL_FBW+rsi*4], edi
    xor ecx, ecx                ; bin
.bin:
    cvtsi2sd xmm0, ecx
    mulsd xmm0, [c_binhz]
    movapd xmm1, xmm0
    subsd xmm0, xmm2
    movapd xmm7, xmm3
    subsd xmm7, xmm2
    divsd xmm0, xmm7            ; rising edge
    movapd xmm7, xmm4
    subsd xmm7, xmm1
    movapd xmm1, xmm4
    subsd xmm1, xmm3
    divsd xmm7, xmm1            ; falling edge
    minsd xmm0, xmm7
    maxsd xmm0, [c_zero]
    mulsd xmm0, xmm6
    ucomisd xmm0, [c_zero]
    jbe .nextbin
    cmp word [rbx+MEL_FBN+rsi*2], 0
    jne .more
    mov [rbx+MEL_FB0+rsi*2], cx
.more:
    inc word [rbx+MEL_FBN+rsi*2]
    cvtsd2ss xmm0, xmm0
    movss [rbx+MEL_W+rdi*4], xmm0
    inc edi
.nextbin:
    inc ecx
    cmp ecx, 257
    jb .bin
    inc esi
    cmp esi, MEL_N
    jb .filter
    mov rcx, [rsp+32]
    call mem_free
    movdqu xmm6, [rsp+64]
    movdqu xmm7, [rsp+80]
    add rsp, 96
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = MEL, rdx = the frame's first sample (f32), r8d = how many of its 400 samples
; exist (zeros after), r9 = f32[257] for the power spectrum
global mel_power
mel_power:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 2048 + 32 + 48
    movdqu [rsp+32+2048], xmm6      ; both callee saved on win64
    movdqu [rsp+32+2048+16], xmm7
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r9
    ; windowed and zero padded, as 256 complex pairs (even, odd), bit reversed
    lea r10, [rsp+32]
    xor ecx, ecx
.load:
    xorps xmm0, xmm0
    cmp ecx, r8d
    jae .put
    cmp ecx, MEL_WINLEN
    jae .put
    movss xmm0, [rsi+rcx*4]
    mulss xmm0, [rbx+MEL_WIN+rcx*4]
.put:
    ; sample n goes to complex m = n / 2, re if even, im if odd, at slot rev[m]
    mov eax, ecx
    shr eax, 1
    movzx eax, word [rbx+MEL_REV+rax*2]
    mov edx, ecx
    and edx, 1
    lea rax, [rax*2+rdx]
    movss [r10+rax*4], xmm0
    inc ecx
    cmp ecx, 512
    jb .load
    ; radix-2 butterflies: len = 2, 4, ... 256
    mov r8d, 2                  ; len
.stage:
    mov r9d, r8d
    shr r9d, 1                  ; half
    mov eax, 256
    xor edx, edx
    div r8d
    mov r11d, eax               ; twiddle stride
    xor ecx, ecx                ; group start
.group:
    xor edx, edx                ; j
.bf:
    mov eax, edx
    imul eax, r11d
    movss xmm4, [rbx+MEL_TWC+rax*4]     ; w = cos - i sin
    movss xmm5, [rbx+MEL_TWS+rax*4]
    lea eax, [rcx+rdx]
    lea r12, [r10+rax*8]        ; a[i+j]
    add eax, r9d
    lea r13, [r10+rax*8]        ; a[i+j+half]
    movss xmm0, [r13]           ; v = a[..+half] * w
    movss xmm1, [r13+4]
    movaps xmm2, xmm0
    mulss xmm2, xmm4
    movaps xmm3, xmm1
    mulss xmm3, xmm5
    addss xmm2, xmm3            ; re: vr c + vi s
    mulss xmm1, xmm4
    mulss xmm0, xmm5
    subss xmm1, xmm0            ; im: vi c - vr s
    movss xmm0, [r12]
    movss xmm3, [r12+4]
    movaps xmm6, xmm0
    addss xmm6, xmm2
    movss [r12], xmm6
    subss xmm0, xmm2
    movss [r13], xmm0
    movaps xmm6, xmm3
    addss xmm6, xmm1
    movss [r12+4], xmm6
    subss xmm3, xmm1
    movss [r13+4], xmm3
    inc edx
    cmp edx, r9d
    jb .bf
    add ecx, r8d
    cmp ecx, 256
    jb .group
    add r8d, r8d
    cmp r8d, 256
    jbe .stage
    ; the split: X[k] = E[k] + e^(-2 pi i k/512) O[k], from Z[k] and conj(Z[256-k])
    xor ecx, ecx
.split:
    mov eax, ecx
    and eax, 255
    movss xmm0, [r10+rax*8]     ; Z[k]
    movss xmm1, [r10+rax*8+4]
    mov eax, 256
    sub eax, ecx
    and eax, 255
    movss xmm2, [r10+rax*8]     ; Z[256-k], conjugated below
    movss xmm3, [r10+rax*8+4]
    ; E = (Z + conj(Zm)) / 2 = ((zr + mr)/2, (zi - mi)/2)
    movaps xmm4, xmm0
    addss xmm4, xmm2
    mulss xmm4, [c_halff]
    movaps xmm5, xmm1
    subss xmm5, xmm3
    mulss xmm5, [c_halff]
    ; D = Z - conj(Zm) = (zr - mr, zi + mi);  O = D / 2i = (di/2, -dr/2)
    subss xmm0, xmm2            ; dr
    addss xmm1, xmm3            ; di
    mulss xmm1, [c_halff]       ; or
    mulss xmm0, [c_halff]
    xorps xmm6, xmm6
    subss xmm6, xmm0            ; oi
    ; W O, W = c - i s: (c or + s oi, c oi - s or)
    movss xmm2, [rbx+MEL_RC+rcx*4]
    movss xmm3, [rbx+MEL_RS+rcx*4]
    movaps xmm0, xmm1
    mulss xmm0, xmm2
    movaps xmm7, xmm6
    mulss xmm7, xmm3
    addss xmm0, xmm7
    addss xmm4, xmm0            ; Xr
    mulss xmm6, xmm2
    mulss xmm1, xmm3
    subss xmm6, xmm1
    addss xmm5, xmm6            ; Xi
    mulss xmm4, xmm4
    mulss xmm5, xmm5
    addss xmm4, xmm5
    movss [rdi+rcx*4], xmm4
    inc ecx
    cmp ecx, 257
    jb .split
    movdqu xmm6, [rsp+32+2048]
    movdqu xmm7, [rsp+32+2048+16]
    add rsp, 2048 + 32 + 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = MEL, rdx = f32[257] power spectrum, r8 = f32[80] out: ln(mel + 1e-6)
global mel_apply
mel_apply:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
    xor r12d, r12d
.f:
    movzx ecx, word [rbx+MEL_FB0+r12*2]
    movzx edx, word [rbx+MEL_FBN+r12*2]
    mov eax, [rbx+MEL_FBW+r12*4]
    lea r8, [rbx+MEL_W+rax*4]
    xorps xmm0, xmm0
    test edx, edx
    jz .sum
.w:
    movss xmm1, [r8]
    mulss xmm1, [rsi+rcx*4]
    addss xmm0, xmm1
    add r8, 4
    inc ecx
    dec edx
    jnz .w
.sum:
    addss xmm0, [c_eps]
    cvtss2sd xmm0, xmm0
    call math_log
    cvtsd2ss xmm0, xmm0
    movss [rdi+r12*4], xmm0
    inc r12d
    cmp r12d, MEL_N
    jb .f
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = MEL, rdx = f32 samples, r8 = how many, r9 = out, f32[frames][80].
; rax = frames = ceil(n / 160)
global mel_run
mel_run:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 1064               ; the power spectrum at +32
    mov rbx, rcx
    mov rsi, rdx
    mov r12, r8
    mov rdi, r9
    lea rax, [r12+MEL_HOP-1]
    xor edx, edx
    mov ecx, MEL_HOP
    div rcx
    mov r13, rax                ; frames
    xor r14d, r14d
.frame:
    cmp r14, r13
    jae .done
    imul rax, r14, MEL_HOP
    mov r8, r12
    sub r8, rax                 ; samples left from here on
    mov ecx, MEL_WINLEN
    cmp r8, rcx
    cmova r8, rcx
    lea rdx, [rsi+rax*4]
    mov rcx, rbx
    lea r9, [rsp+32]
    call mel_power
    mov rcx, rbx
    lea rdx, [rsp+32]
    mov r8, rdi
    call mel_apply
    add rdi, MEL_N * 4
    inc r14
    jmp .frame
.done:
    mov rax, r13
    add rsp, 1064
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
