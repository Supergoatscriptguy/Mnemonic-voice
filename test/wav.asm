; stage 2 test: wav parsing and writing, on hand-made files and round trips
; uses: audio\wav
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "audio/audio.inc"

section .rdata
align 4
; pcm16 stereo, with an odd-sized LIST chunk (so a pad byte) in front of fmt
w16:    db 'RIFF'
        dd w16e - w16 - 8
        db 'WAVE', 'LIST'
        dd 3
        db 'abc', 0
        db 'fmt '
        dd 16
        dw 1, 2
        dd 22050, 22050*4
        dw 4, 16
        db 'data'
        dd 12
        dw 0x7fff, 0x8000, 1, 0xffff, 0, 0x4000
w16e:
x16:    dd 0.999969482421875, -1.0, 0.000030517578125, -0.000030517578125, 0.0, 0.5

; 24 bit mono as WAVE_FORMAT_EXTENSIBLE, odd data size
w24:    db 'RIFF'
        dd w24e - w24 - 8
        db 'WAVE', 'fmt '
        dd 40
        dw 0xfffe, 1
        dd 48000, 48000*3
        dw 3, 24
        dw 22, 24
        dd 4
        dd 1                    ; KSDATAFORMAT_SUBTYPE_PCM
        dw 0, 0x10
        db 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71
        db 'data'
        dd 9
        db 0xff, 0xff, 0x7f, 0, 0, 0x80, 1, 0, 0, 0
w24e:
x24:    dd 0.99999988079071044921875, -1.0, 1.1920928955078125e-7

; float with a fact chunk, and the 0xffffffff sizes a streaming writer leaves
wf:     db 'RIFF'
        dd 0xffffffff
        db 'WAVE', 'fmt '
        dd 18
        dw 3, 1
        dd 16000, 64000
        dw 4, 32, 0
        db 'fact'
        dd 4, 2
        db 'data'
        dd 0xffffffff
xf:     dd 0.25, -0.75
wfe:

; 8 bit pcm, which we don't read
w8:     db 'RIFF'
        dd w8e - w8 - 8
        db 'WAVE', 'fmt '
        dd 16
        dw 1, 1
        dd 8000, 8000
        dw 1, 8
        db 'data'
        dd 2
        db 0x80, 0xff
w8e:

; fmt but no data
wnd:    db 'RIFF'
        dd wnde - wnd - 8
        db 'WAVE', 'fmt '
        dd 16
        dw 1, 1
        dd 8000, 16000
        dw 2, 16
wnde:

; on the way out to pcm16: clipping, and halves round to even
clipin:  dd 1.5, -1.5, 0.99999, -1.00001, 0.0000152587890625, 0.0000457763671875, -0.0000152587890625
clipout: dd 0.999969482421875, -1.0, 0.999969482421875, -1.0, 0.0, 0.00006103515625, 0.0

s15     dd 0x38000000
c_two   dd 2.0
c_one   dd 1.0
tmpwav  db "build\wavtest.wav", 0
nowav   db "build\no such file.wav", 0

section .bss
alignb 8
clip    resb CL_SIZE
clip2   resb CL_SIZE
ra      resb RNG_SIZE
buf     resd 2000

section .text

; rcx = a, rdx = b, r8 = dwords. eax = 1 if they're the same bits
same:
    push rsi
    push rdi
    mov rsi, rcx
    mov rdi, rdx
    mov rcx, r8
    xor eax, eax
    repe cmpsd
    sete al
    pop rdi
    pop rsi
    ret

%macro parse 2
    lea rcx, [%1]
    mov edx, %2 - %1
    lea r8, [clip]
    call wav_parse
%endmacro

; clip's samples vs n expected dwords
%macro samples 2+
    mov rcx, [clip+CL_DATA]
    lea rdx, [%1]
    mov r8d, %2
    call same
    cmp eax, 1
%endmacro

; write clip, read it back into clip2, compare
%macro roundtrip 2+
    lea rcx, [tmpwav]
    lea rdx, [clip]
    mov r8d, %1
    call wav_write
    cmp eax, 1
    check e, %2
    lea rcx, [tmpwav]
    lea rdx, [clip2]
    call wav_read
    mov ebx, eax
    mov rax, [clip+CL_FRAMES]
    cmp rax, [clip2+CL_FRAMES]
    jne %%bad
    mov eax, [clip+CL_RATE]
    cmp eax, [clip2+CL_RATE]
    jne %%bad
    mov eax, [clip+CL_CH]
    cmp eax, [clip2+CL_CH]
    jne %%bad
    cmp ebx, 1
%%bad:
    check e, "  read back, same rate, channels, frames"
%endmacro

global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    call lib_init

    say "parsing", 13, 10
    parse w16, w16e
    cmp eax, 1
    check e, "pcm16 stereo behind an odd LIST chunk"
    cmp dword [clip+CL_RATE], 22050
    check e, "  22050 Hz"
    cmp dword [clip+CL_CH], 2
    check e, "  2 channels"
    cmp qword [clip+CL_FRAMES], 3
    check e, "  3 frames"
    samples x16, 6
    check e, "  32767 -32768 1 -1 0 16384 come out exact"

    parse w24, w24e
    cmp eax, 1
    check e, "24 bit WAVE_FORMAT_EXTENSIBLE, odd data size"
    cmp dword [clip+CL_BITS], 24
    check e, "  24 bits"
    cmp qword [clip+CL_FRAMES], 3
    check e, "  3 frames"
    samples x24, 3
    check e, "  max, min, 1 lsb exact"

    parse wf, wfe
    cmp eax, 1
    check e, "float with a fact chunk and 0xffffffff sizes"
    cmp dword [clip+CL_FLOAT], 1
    check e, "  flagged float"
    cmp qword [clip+CL_FRAMES], 2
    check e, "  2 frames (size clamped to the file)"
    samples xf, 2
    check e, "  0.25 -0.75"

    parse w8, w8e
    cmp eax, 0
    check e, "8 bit pcm is refused"
    parse wnd, wnde
    cmp eax, 0
    check e, "no data chunk is refused"
    parse x16, w24
    cmp eax, 0
    check e, "not riff is refused"
    lea rcx, [nowav]
    lea rdx, [clip]
    call wav_read
    cmp eax, 0
    check e, "missing file gives 0"

    say "round trips", 13, 10
    ; random pcm16 values, stereo
    lea rcx, [ra]
    mov edx, 7
    call rng_seed
    lea rsi, [buf]
    xor ebx, ebx
.r16:
    lea rcx, [ra]
    call rng_next
    movsx eax, ax
    cvtsi2ss xmm0, eax
    mulss xmm0, [s15]
    movss [rsi+rbx*4], xmm0
    inc ebx
    cmp ebx, 2000
    jb .r16
    mov [clip+CL_DATA], rsi
    mov qword [clip+CL_FRAMES], 1000
    mov dword [clip+CL_RATE], 44100
    mov dword [clip+CL_CH], 2
    roundtrip 16, "write 1000 stereo frames of random pcm16"
    mov rcx, [clip2+CL_DATA]
    mov rdx, rsi
    mov r8d, 2000
    call same
    cmp eax, 1
    check e, "  pcm16 -> f32 -> pcm16 -> f32 is bit exact"
    mov rcx, [clip2+CL_DATA]
    call mem_free

    ; random floats in -1..1, mono
    xor ebx, ebx
.rf:
    lea rcx, [ra]
    call rng_float
    cvtsd2ss xmm0, xmm0
    mulss xmm0, [c_two]
    subss xmm0, [c_one]
    movss [rsi+rbx*4], xmm0
    inc ebx
    cmp ebx, 2000
    jb .rf
    mov qword [clip+CL_FRAMES], 2000
    mov dword [clip+CL_RATE], 48000
    mov dword [clip+CL_CH], 1
    roundtrip 32, "write 2000 random floats"
    mov rcx, [clip2+CL_DATA]
    mov rdx, rsi
    mov r8d, 2000
    call same
    cmp eax, 1
    check e, "  float is bit exact"
    cmp dword [clip2+CL_FLOAT], 1
    check e, "  and reads back as float"
    mov rcx, [clip2+CL_DATA]
    call mem_free

    lea rax, [clipin]
    mov [clip+CL_DATA], rax
    mov qword [clip+CL_FRAMES], 7
    mov dword [clip+CL_RATE], 16000
    roundtrip 16, "write out-of-range and half-lsb values as pcm16"
    mov rcx, [clip2+CL_DATA]
    lea rdx, [clipout]
    mov r8d, 7
    call same
    cmp eax, 1
    check e, "  clipped to -1 / 32767, halves to even"

    jmp t_done
