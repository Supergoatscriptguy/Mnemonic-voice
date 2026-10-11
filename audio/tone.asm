; writes test sounds to samples\: a bell-ish chord (pcm16) and a log sine sweep (float)
; uses: audio\wav
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

RATE    equ 48000
NCHIME  equ RATE * 5 / 2
NSWEEP  equ RATE * 4

section .rdata
align 8
notes   dq 523.25, 659.25, 783.99, 1046.50     ; C5 E5 G5 C6
c_rate  dq 48000.0
c_2pi   dq 6.283185307179586
c_gap   dq 0.15                 ; seconds between notes
c_mtau  dq -0.6                 ; decay of each note
c_mtau2 dq -0.25                ; its octave partial dies faster
c_amp   dq 0.18
c_amp2  dq 0.3
c_att   dq 200.0                ; 5 ms attack so it doesn't click
c_one   dq 1.0
c_half  dq 0.5
c_f1    dq 20.0
c_f2    dq 20000.0
c_len   dq 4.0
c_fade  dq 100.0                ; 10 ms fades
p_dir   db "samples", 0
p_chime db "samples\chime.wav", 0
p_sweep db "samples\sweep.wav", 0
p_err   db "couldn't write a wav into samples\", 0

section .bss
alignb 8
clip    resb CL_SIZE

section .text
global start
start:
    push rbx
    push rsi
    push r12
    push r13
    sub rsp, 104
    ; locals: t [rsp+32], acc +40, dt +48, ramp +56, e +64, ph +72, part +80, k +88
    call lib_init
    lea rcx, [p_dir]
    call make_dir

    mov ecx, NCHIME * 4
    call mem_alloc
    mov rbx, rax
    lea rsi, [notes]
    xor r12d, r12d
.chime:
    cvtsi2sd xmm0, r12
    divsd xmm0, [c_rate]
    movsd [rsp+32], xmm0
    xorpd xmm0, xmm0
    movsd [rsp+40], xmm0
    xor r13d, r13d
.note:
    cvtsi2sd xmm0, r13
    mulsd xmm0, [c_gap]
    movsd xmm1, [rsp+32]
    subsd xmm1, xmm0
    xorpd xmm2, xmm2
    comisd xmm1, xmm2
    jb .later
    movsd [rsp+48], xmm1
    mulsd xmm1, [c_att]
    minsd xmm1, [c_one]
    movsd [rsp+56], xmm1
    ; e^(-dt/tau) sin(2 pi f dt) + a2 e^(-dt/tau2) sin(4 pi f dt)
    movsd xmm0, [rsp+48]
    divsd xmm0, [c_mtau]
    call math_exp
    movsd [rsp+64], xmm0
    movsd xmm0, [c_2pi]
    mulsd xmm0, [rsi+r13*8]
    mulsd xmm0, [rsp+48]
    movsd [rsp+72], xmm0
    call math_sin
    mulsd xmm0, [rsp+64]
    movsd [rsp+80], xmm0
    movsd xmm0, [rsp+48]
    divsd xmm0, [c_mtau2]
    call math_exp
    movsd [rsp+64], xmm0
    movsd xmm0, [rsp+72]
    addsd xmm0, xmm0
    call math_sin
    mulsd xmm0, [rsp+64]
    mulsd xmm0, [c_amp2]
    addsd xmm0, [rsp+80]
    mulsd xmm0, [rsp+56]
    mulsd xmm0, [c_amp]
    addsd xmm0, [rsp+40]
    movsd [rsp+40], xmm0
.later:
    inc r13d
    cmp r13d, 4
    jb .note
    cvtsd2ss xmm0, [rsp+40]
    movss [rbx+r12*4], xmm0
    inc r12d
    cmp r12d, NCHIME
    jb .chime

    mov [clip+CL_DATA], rbx
    mov qword [clip+CL_FRAMES], NCHIME
    mov dword [clip+CL_RATE], RATE
    mov dword [clip+CL_CH], 1
    lea rcx, [p_chime]
    lea rdx, [clip]
    mov r8d, 16
    call wav_write
    test eax, eax
    jz .fail
    say "wrote samples\chime.wav   C5 E5 G5 C6 bells, 2.5 s, 48 kHz pcm16", 13, 10
    mov rcx, rbx
    call mem_free

    ; log sweep 20 Hz - 20 kHz: phase = 2 pi f1 T / K (e^(t K / T) - 1), K = ln(f2/f1)
    mov ecx, NSWEEP * 4
    call mem_alloc
    mov rbx, rax
    movsd xmm0, [c_f2]
    divsd xmm0, [c_f1]
    call math_log
    movsd [rsp+88], xmm0                ; K
    movsd xmm1, [c_2pi]
    mulsd xmm1, [c_f1]
    mulsd xmm1, [c_len]
    divsd xmm1, xmm0
    movsd [rsp+80], xmm1                ; 2 pi f1 T / K
    xor r12d, r12d
.sweep:
    cvtsi2sd xmm0, r12
    divsd xmm0, [c_rate]
    movsd [rsp+32], xmm0
    ; fade = min(1, t / 10ms, (T - t) / 10ms)
    movsd xmm1, [c_len]
    subsd xmm1, xmm0
    minsd xmm1, xmm0
    mulsd xmm1, [c_fade]
    minsd xmm1, [c_one]
    movsd [rsp+56], xmm1
    mulsd xmm0, [rsp+88]
    divsd xmm0, [c_len]
    call math_exp
    subsd xmm0, [c_one]
    mulsd xmm0, [rsp+80]
    call math_sin
    mulsd xmm0, [c_half]
    mulsd xmm0, [rsp+56]
    cvtsd2ss xmm0, xmm0
    movss [rbx+r12*4], xmm0
    inc r12d
    cmp r12d, NSWEEP
    jb .sweep

    mov [clip+CL_DATA], rbx
    mov qword [clip+CL_FRAMES], NSWEEP
    lea rcx, [p_sweep]
    lea rdx, [clip]
    mov r8d, 32
    call wav_write
    test eax, eax
    jz .fail
    say "wrote samples\sweep.wav   20 Hz - 20 kHz log sweep, 4 s, 48 kHz float, -6 dBFS", 13, 10
    call con_restore
    xor ecx, ecx
    call ExitProcess
.fail:
    lea rcx, [p_err]
    call fatal
