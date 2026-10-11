; play file.wav: plays it on the default output device (wasapi, shared, event driven)
; uses: audio\wav audio\dev
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

section .bss
alignb 8
clip    resb CL_SIZE
dev     resb DV_SIZE

section .text

; gives the device as much of the clip as it'll take.
; r12 = frames handed over so far, r13 = all frames, r14 = channels, r15 = samples
feed:
    push rbx
    sub rsp, 32
    lea rcx, [dev]
    call dev_space
    mov rbx, r13
    sub rbx, r12
    cmp rbx, rax
    cmova rbx, rax
    test rbx, rbx
    jz .out
    mov rdx, r12
    imul rdx, r14
    lea rdx, [r15+rdx*4]
    lea rcx, [dev]
    mov r8d, ebx
    call dev_write
    add r12, rbx
.out:
    add rsp, 32
    pop rbx
    ret

; waits a period, bails out if the device has gone quiet
period:
    sub rsp, 40
    lea rcx, [dev]
    call dev_wait
    test eax, eax
    jnz .dead
    add rsp, 40
    ret
.dead:
    say "the device stopped asking for audio", 13, 10
    mov ecx, 1
    call ExitProcess

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    call lib_init
    cmp qword [argc], 2
    jae .go
    say "usage: play file.wav", 13, 10
    mov ecx, 1
    call ExitProcess
.go:
    lea rax, [argv]
    mov rcx, [rax+8]
    lea rdx, [clip]
    call wav_read
    test eax, eax
    jnz .read
    say "can't read that wav", 13, 10
    mov ecx, 1
    call ExitProcess
.read:
    xor ecx, ecx
    lea rdx, [dev]
    mov r8d, 50
    call dev_open
    test eax, eax
    jnz .opened
    say "the output device doesn't mix in float32, not handled yet", 13, 10
    mov ecx, 1
    call ExitProcess
.opened:
    lea rcx, [dev]
    call dev_show
    mov eax, [clip+CL_RATE]
    cmp eax, [dev+DV_RATE]
    je .samerate
    say "the file is "
    mov ecx, [clip+CL_RATE]
    call print_dec
    say " Hz and the device runs at "
    mov ecx, [dev+DV_RATE]
    call print_dec
    say " Hz, that needs the resampler", 13, 10
    mov ecx, 1
    call ExitProcess

.samerate:
    ; lay the clip out in the device's channels: mono goes to all of them,
    ; extra file channels get dropped, missing ones stay silent
    mov r13, [clip+CL_FRAMES]
    mov r14d, [dev+DV_CH]
    mov rcx, r13
    imul rcx, r14
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov r15, rax
    mov rsi, [clip+CL_DATA]
    mov rdi, rax
    mov r9d, [clip+CL_CH]
    xor r8d, r8d
    test r13, r13
    jz .laidout
.frame:
    xor ecx, ecx
.ch:
    xor eax, eax
    mov edx, ecx
    cmp r9d, 1
    jne .multi
    xor edx, edx
.multi:
    cmp edx, r9d
    jae .put
    mov eax, [rsi+rdx*4]
.put:
    mov [rdi], eax
    add rdi, 4
    inc ecx
    cmp ecx, r14d
    jb .ch
    lea rsi, [rsi+r9*4]
    inc r8
    cmp r8, r13
    jb .frame
.laidout:

    xor r12d, r12d
    call feed                   ; fill the buffer before starting, or the first period glitches
    lea rcx, [dev]
    call dev_start
    call time_now
    mov [rsp+32], rax
.more:
    cmp r12, r13
    jae .drain
    call period
    call feed
    jmp .more
.drain:
    call period
    lea rcx, [dev]
    call dev_space
    cmp eax, [dev+DV_BUF]
    jb .drain
    call period                 ; one more period, the engine reads ahead of the speaker
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    say "played "
    cvtsi2sd xmm0, r13
    cvtsi2sd xmm1, dword [dev+DV_RATE]
    divsd xmm0, xmm1
    mov edx, 3
    call print_fixed
    say " s of audio in "
    movsd xmm0, [rsp+40]
    mov edx, 3
    call print_fixed
    say " s", 13, 10
    lea rcx, [dev]
    call dev_stop
    lea rcx, [dev]
    call dev_close
    call con_restore
    xor ecx, ecx
    call ExitProcess
