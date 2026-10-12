; wav files in and out. a decoded clip is always f32 samples, interleaved, -1..1
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

section .rdata
align 4
s15     dd 0x38000000           ; 2^-15
s23     dd 0x34000000           ; 2^-23
s31     dd 0x30000000           ; 2^-31
p15     dd 32768.0
lo16    dd -32768.0
hi16    dd 32767.0

section .text

; rcx = bytes, rdx = size, r8 = clip. reads just the header: rate, channels,
; bits, frames, and CL_DATA = the raw samples in place. eax = 1 if it's a wav we
; can read: pcm 16/24/32 or float32, plain or WAVE_FORMAT_EXTENSIBLE
global wav_head
wav_head:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov rsi, rcx
    lea rdi, [rcx+rdx]
    mov r12, r8
    xor r13d, r13d              ; fmt chunk
    xor r14d, r14d              ; data chunk
    xor r15d, r15d              ; and its size
    cmp rdx, 12
    jb .no
    cmp dword [rsi], 'RIFF'
    jne .no
    cmp dword [rsi+8], 'WAVE'
    jne .no
    add rsi, 12
.chunk:
    lea rax, [rsi+8]
    cmp rax, rdi
    ja .scanned
    mov eax, [rsi]
    mov ecx, [rsi+4]
    add rsi, 8
    mov rdx, rdi
    sub rdx, rsi                ; what's left
    cmp eax, 'fmt '
    jne .notfmt
    cmp ecx, 16
    jb .no
    cmp rcx, rdx
    ja .no
    mov r13, rsi
    jmp .next
.notfmt:
    cmp eax, 'data'
    jne .next
    cmp rcx, rdx                ; streamed files leave 0xffffffff (or junk) here
    cmova rcx, rdx
    mov r14, rsi
    mov r15, rcx
    test r13, r13
    jnz .scanned                ; fmt already seen, ignore whatever trails the data
.next:
    inc rcx                     ; chunks are padded to an even size
    and rcx, -2
    add rsi, rcx
    jmp .chunk

.scanned:
    test r13, r13
    jz .no
    test r14, r14
    jz .no
    movzx eax, word [r13]       ; format tag
    cmp eax, 0xfffe
    jne .tag
    cmp dword [r13-4], 26       ; extensible: the real tag is the subformat guid's first word
    jb .no
    movzx eax, word [r13+24]
.tag:
    movzx ebx, word [r13+14]    ; bits per sample
    cmp eax, 3
    je .isfloat
    cmp eax, 1
    jne .no
    cmp ebx, 16
    je .known
    cmp ebx, 24
    je .known
    cmp ebx, 32
    je .known
    jmp .no
.isfloat:
    cmp ebx, 32
    jne .no
.known:
    mov [r12+CL_BITS], ebx
    xor ecx, ecx
    cmp eax, 3
    sete cl
    mov [r12+CL_FLOAT], ecx
    movzx ecx, word [r13+2]
    test ecx, ecx
    jz .no
    mov [r12+CL_CH], ecx
    mov eax, [r13+4]
    mov [r12+CL_RATE], eax
    shr ebx, 3                  ; bytes per sample
    imul ecx, ebx               ; per frame
    mov rax, r15
    xor edx, edx
    div rcx
    mov [r12+CL_FRAMES], rax
    mov [r12+CL_DATA], r14
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = bytes, rdx = size, r8 = clip to fill. eax = 1 if it's a wav we can read:
; pcm 16/24/32 or float32, plain or WAVE_FORMAT_EXTENSIBLE. CL_DATA gets
; mem_alloc'd, mem_free it
global wav_parse
wav_parse:
    push rbx
    push rsi
    push rdi
    push r12
    push r15
    sub rsp, 32
    mov r12, r8
    call wav_head
    test eax, eax
    jz .no
    mov rsi, [r12+CL_DATA]
    mov ebx, [r12+CL_BITS]
    shr ebx, 3                  ; bytes per sample
    mov rax, [r12+CL_FRAMES]
    mov ecx, [r12+CL_CH]
    imul rax, rcx
    mov r15, rax                ; samples
    lea rcx, [rax*4+64]
    call mem_alloc
    mov [r12+CL_DATA], rax
    mov rdi, rax
    test r15, r15
    jz .yes
    cmp dword [r12+CL_FLOAT], 0
    jne .f32
    cmp ebx, 2
    je .i16
    cmp ebx, 3
    je .i24
.i32:
    cvtsi2ss xmm0, dword [rsi]
    mulss xmm0, [s31]
    movss [rdi], xmm0
    add rsi, 4
    add rdi, 4
    dec r15
    jnz .i32
    jmp .yes
.i24:
    movzx eax, word [rsi]
    movsx edx, byte [rsi+2]
    shl edx, 16
    or eax, edx
    cvtsi2ss xmm0, eax
    mulss xmm0, [s23]
    movss [rdi], xmm0
    add rsi, 3
    add rdi, 4
    dec r15
    jnz .i24
    jmp .yes
.i16:
    movsx eax, word [rsi]
    cvtsi2ss xmm0, eax
    mulss xmm0, [s15]
    movss [rdi], xmm0
    add rsi, 2
    add rdi, 4
    dec r15
    jnz .i16
    jmp .yes
.f32:
    mov rcx, r15
    rep movsd
.yes:
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop r15
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = path, rdx = clip. eax = 1 if read, 0 if there's no file or it's not a
; wav we can read
global wav_read
wav_read:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rdx
    call file_read_all
    test rax, rax
    jz .out
    mov rsi, rax
    mov rcx, rax
    mov r8, rbx
    call wav_parse
    mov ebx, eax
    mov rcx, rsi
    call mem_free
    mov eax, ebx
.out:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = path, rdx = clip, r8d = 16 (pcm) or 32 (float). eax = 1 if it all got written.
; pcm16 rounds x * 32768 to nearest and clips, so reading pcm16 and writing it
; back gives the same bytes
global wav_write
wav_write:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov r12, rcx
    mov r13, rdx
    mov r14d, r8d
    mov rax, [r13+CL_FRAMES]
    mov ecx, [r13+CL_CH]
    imul rax, rcx
    mov rbx, rax                ; samples
    mov ecx, r14d
    shr ecx, 3
    imul rax, rcx
    mov rsi, rax                ; data bytes
    lea rcx, [rax+64]
    call mem_alloc
    mov rdi, rax

    mov dword [rdi], 'RIFF'
    mov dword [rdi+8], 'WAVE'
    mov dword [rdi+12], 'fmt '
    mov ecx, [r13+CL_CH]
    mov [rdi+22], cx
    mov eax, [r13+CL_RATE]
    mov [rdi+24], eax
    mov edx, r14d
    shr edx, 3
    imul ecx, edx               ; block align
    mov [rdi+32], cx
    imul eax, ecx
    mov [rdi+28], eax           ; bytes per second
    mov [rdi+34], r14w
    cmp r14d, 16
    jne .fhdr
    mov dword [rdi+16], 16
    mov word [rdi+20], 1
    mov dword [rdi+36], 'data'
    mov [rdi+40], esi
    lea eax, [rsi+36]
    mov [rdi+4], eax
    mov r8d, 44
    jmp .body
.fhdr:
    ; non-pcm formats are supposed to carry cbSize and a fact chunk
    mov dword [rdi+16], 18
    mov word [rdi+20], 3
    mov word [rdi+36], 0
    mov dword [rdi+38], 'fact'
    mov dword [rdi+42], 4
    mov rax, [r13+CL_FRAMES]
    mov [rdi+46], eax
    mov dword [rdi+50], 'data'
    mov [rdi+54], esi
    lea eax, [rsi+50]
    mov [rdi+4], eax
    mov r8d, 58

.body:
    lea rdx, [rdi+r8]
    add rsi, r8                 ; whole file
    mov rcx, [r13+CL_DATA]
    test rbx, rbx
    jz .write
    cmp r14d, 16
    jne .copy
    movss xmm1, [p15]
    movss xmm2, [lo16]
    movss xmm3, [hi16]
.pcm:
    movss xmm0, [rcx]
    mulss xmm0, xmm1
    maxss xmm0, xmm2
    minss xmm0, xmm3
    cvtss2si eax, xmm0          ; nearest, ties to even
    mov [rdx], ax
    add rcx, 4
    add rdx, 2
    dec rbx
    jnz .pcm
    jmp .write
.copy:
    mov eax, [rcx]
    mov [rdx], eax
    add rcx, 4
    add rdx, 4
    dec rbx
    jnz .copy

.write:
    mov rcx, r12
    call file_create
    cmp rax, -1
    je .fail
    mov rbx, rax
    mov rcx, rax
    mov rdx, rdi
    mov r8, rsi
    call file_write
    mov esi, eax
    mov rcx, rbx
    call file_close
    jmp .done
.fail:
    xor esi, esi
.done:
    mov rcx, rdi
    call mem_free
    mov eax, esi
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = clip. averages its channels down to one, in place
global clip_mono
clip_mono:
    mov r8d, [rcx+CL_CH]
    cmp r8d, 1
    jbe .done
    mov rdx, [rcx+CL_DATA]
    mov r9, rdx
    mov r10, [rcx+CL_FRAMES]
    cvtsi2ss xmm2, r8d
    mov eax, 0x3f800000         ; 1.0
    movd xmm3, eax
    divss xmm3, xmm2
    test r10, r10
    jz .one
.frame:
    xorps xmm0, xmm0
    mov eax, r8d
.ch:
    addss xmm0, [rdx]
    add rdx, 4
    dec eax
    jnz .ch
    mulss xmm0, xmm3
    movss [r9], xmm0
    add r9, 4
    dec r10
    jnz .frame
.one:
    mov dword [rcx+CL_CH], 1
.done:
    ret
