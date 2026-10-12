; live PRESET [ckpt=PATH] [secs=30] [every=400] [chunk=0]
; dictation, live: the default mic through the speech model while you talk. every
; `every` ms the utterance so far goes through the encoder (all of it, again: the
; simple way, before the encoder keeps state between chunks) and its greedy
; transcript is redrawn in place. after secs seconds the line is kept and a new one
; starts. chunk=N attends in chunks of N encoder frames, as streaming would.
; ctrl+c stops
; uses: gpu\cuda gpu\kernels asr\model asr\ckpt asr\score audio\dev audio\resample audio\mel tokenizer\pretok tokenizer\bpe tokenizer\tok
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "audio/audio.inc"
%include "tokenizer/tok.inc"
%include "asr/model.inc"
%include "asr/train.inc"

extern ctc_greedy

section .rdata
s_asr    db "asr\", 0
s_cfg    db ".cfg", 0
s_ckdir  db "checkpoints\", 0
s_star   db "\step_*.ckpt", 0
s_tok    db "datasets\sptok.bin", 0
s_melete db "melete", 0
k_run    db "run", 0
k_ckpt   db "ckpt", 0
k_secs   db "secs", 0
k_every  db "every", 0
k_chunk  db "chunk", 0
e_preset db "no such preset (asr\<name>.cfg)", 0
e_tok    db "can't load the tokenizer", 0
e_ckpt   db "no checkpoint to listen with", 0
e_mic    db "the mic doesn't give float32", 0
align 4
c_inv    dd 1.0

section .bss
alignb 8
run      resq 1
cfgpath  resb 512
path     resb 512
pattern  resb 512
hdr      resb CK_SIZE
dev      resb DV_SIZE
rs       resb RS_SIZE
mel      resb MEL_SIZE
mb       resb MB_SIZE
secs     resq 1
every    resq 1
chunk    resq 1
maxs     resq 1                 ; 16 kHz samples a line holds
pcm      resq 1                 ; the line so far, 16 kHz mono f32
npcm     resq 1
raw      resq 1                 ; a packet's worth from the device
mono     resq 1
hmel     resq 1                 ; pinned
hbest    resq 1
d_mel    resq 1
d_tl     resq 1
d_dummy  resq 1
d_best   resq 1
since    resq 1                 ; samples since the last redraw
toks     resw 8192
text     resb 1 << 15
tl       resd 1

section .text

; rcx = arg. eax = 1 if it's not a key=value one
plain:
    xor eax, eax
.c:
    mov dl, [rcx]
    test dl, dl
    jz .yes
    cmp dl, '='
    je .no
    inc rcx
    jmp .c
.yes:
    mov eax, 1
.no:
    ret

; asr\<first plain arg>.cfg, then the key=value args on top
preset:
    push rbx
    push rsi
    sub rsp, 40
    lea rsi, [s_melete]
    mov ebx, 1
.a:
    cmp rbx, [argc]
    jae .have
    lea rax, [argv]
    mov rcx, [rax+rbx*8]
    call plain
    test eax, eax
    jz .next
    lea rax, [argv]
    mov rsi, [rax+rbx*8]
    jmp .have
.next:
    inc ebx
    jmp .a
.have:
    mov [run], rsi
    lea rcx, [cfgpath]
    lea rdx, [s_asr]
    call fmt_str
    mov rcx, rax
    mov rdx, rsi
    call fmt_str
    mov rcx, rax
    lea rdx, [s_cfg]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [cfgpath]
    call cfg_load
    test eax, eax
    jnz .loaded
    lea rcx, [e_preset]
    call fatal
.loaded:
    call cfg_args
    add rsp, 40
    pop rsi
    pop rbx
    ret

; the line so far through the model; its transcript redrawn
listen:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    ; log-mel, normalized with the checkpoint's statistics, zeros to T (a multiple
    ; of 1024 mel frames, so B*T3 is a multiple of 128 for the fast matmuls)
    lea rcx, [mel]
    mov rdx, [pcm]
    mov r8, [npcm]
    mov r9, [hmel]
    call mel_run
    mov rbx, rax                ; frames
    test rbx, rbx
    jz .done
    mov rsi, [hmel]
    lea r8, [hdr+CK_MEAN]
    lea r9, [hdr+CK_ISTD]
    xor ecx, ecx
.f:
    cmp rcx, rbx
    jae .fd
    imul rdx, rcx, MEL_N * 4
    add rdx, rsi
    xor eax, eax
.b:
    movss xmm0, [rdx+rax*4]
    subss xmm0, [r8+rax*4]
    mulss xmm0, [r9+rax*4]
    movss [rdx+rax*4], xmm0
    inc eax
    cmp eax, MEL_N
    jb .b
    inc rcx
    jmp .f
.fd:
    lea rdi, [rbx+1023]
    and rdi, -1024              ; T
    imul rcx, rbx, MEL_N * 4
    add rcx, rsi
    mov rax, rdi
    sub rax, rbx
    imul rax, MEL_N
.z:
    test rax, rax
    jz .zd
    mov dword [rcx], 0
    add rcx, 4
    dec rax
    jmp .z
.zd:
    lea eax, [rbx+7]
    shr eax, 3
    mov [tl], eax
    mov rcx, [d_mel]
    mov rdx, [hmel]
    imul r8, rdi, MEL_N * 4
    call gpu_up
    mov rcx, [d_tl]
    lea rdx, [tl]
    mov r8d, 4
    call gpu_up
    mov dword [mb+MB_B], 1
    mov [mb+MB_T], edi
    mov dword [mb+MB_UMAX], 1
    mov eax, edi
    shr eax, 3
    mov [mb+MB_CS], eax         ; full context unless chunk= says
    mov dword [mb+MB_LEFT], 0
    mov rax, [chunk]
    test rax, rax
    jz .ctx
    mov [mb+MB_CS], eax
    mov dword [mb+MB_LEFT], 0xffffffff
.ctx:
    lea rcx, [mb]
    mov edx, FW_LOGITS
    call model_fwd
    mov rcx, [d_best]
    call model_argmax
    mov rcx, [hbest]
    mov rdx, [d_best]
    mov r8d, [tl]
    shl r8, 2
    call gpu_down
    mov rcx, [hbest]
    mov edx, [tl]
    mov r8, [mdl+MD_V]
    dec r8d
    lea r9, [toks]
    call ctc_greedy
    lea rcx, [toks]
    mov rdx, rax
    lea r8, [text]
    call tok_decode
    ; redraw: the last 110 bytes that fit, from a character boundary
    mov rsi, rax
    lea rdi, [text]
    cmp rsi, 110
    jbe .show
    add rdi, rsi
    sub rdi, 110
    mov rsi, 110
.utf:
    mov al, [rdi]
    and al, 0xc0
    cmp al, 0x80
    jne .show
    inc rdi
    dec rsi
    jmp .utf
.show:
    say 13, 27, "[K  "
    mov rcx, rdi
    mov rdx, rsi
    call print
.done:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

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
    call preset
    lea rcx, [k_run]
    mov rdx, [run]
    call cfg_str
    mov [run], rax
    lea rcx, [k_secs]
    mov edx, 30
    call cfg_int
    mov [secs], rax
    lea rcx, [k_every]
    mov edx, 400
    call cfg_int
    imul rax, 16                ; ms -> samples at 16 kHz
    mov [every], rax
    lea rcx, [k_chunk]
    xor edx, edx
    call cfg_int
    mov [chunk], rax
    call ctrlc_init
    call gpu_init
    xor ecx, ecx
    call pool_init
    lea rcx, [s_tok]
    call tok_load
    test eax, eax
    jnz .tok
    lea rcx, [e_tok]
    call fatal
.tok:
    lea rcx, [mel]
    call mel_init

    ; the model, for one utterance of up to secs seconds
    call model_config
    mov rax, [secs]
    imul rax, 100
    add rax, 1023
    and rax, -1024
    mov [mdl+MD_T], rax
    mov qword [mdl+MD_B], 1
    shr rax, 3
    mov [mdl+MD_UMAX], rax
    mov dword [mdl_fast], 1
    mov ecx, 2
    call model_setup
    lea rcx, [k_ckpt]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .newest
    cmp byte [rax], 0
    je .newest
    mov rsi, rax
    lea rdi, [path]
.cp:
    mov al, [rsi]
    mov [rdi], al
    inc rsi
    inc rdi
    test al, al
    jnz .cp
    jmp .load
.newest:
    lea rcx, [pattern]
    lea rdx, [s_ckdir]
    call fmt_str
    mov rcx, rax
    mov rdx, [run]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_star]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [pattern]
    lea rdx, [path]
    call ck_latest
    test eax, eax
    jnz .load
    lea rcx, [e_ckpt]
    call fatal
.load:
    lea rcx, [path]
    lea rdx, [hdr]
    call ck_load
    say "  "
    lea rcx, [path]
    call print_z
    say ", step "
    mov rcx, [hdr+CK_STEP]
    call print_dec
    say 13, 10

    ; buffers
    mov rax, [secs]
    imul rax, 16000
    mov [maxs], rax
    lea rcx, [rax*4+65536]
    call mem_alloc
    mov [pcm], rax
    mov rcx, [mdl+MD_T]
    imul rcx, MEL_N * 4
    mov rbx, rcx
    call gpu_host
    mov [hmel], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_mel], rax
    mov [mb+MB_MEL], rax
    mov ecx, 64
    call gpu_alloc
    mov [d_tl], rax
    mov [mb+MB_TL], rax
    mov ecx, 64
    call gpu_alloc
    mov [d_dummy], rax
    mov [mb+MB_UL], rax
    mov [mb+MB_LAB], rax
    mov [mb+MB_LOSS], rax
    mov rcx, [mdl+MD_T3]
    shl rcx, 2
    mov rbx, rcx
    call gpu_host
    mov [hbest], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_best], rax

    ; the mic
    mov ecx, 1
    lea rdx, [dev]
    mov r8d, 50
    call dev_open
    test eax, eax
    jnz .opened
    lea rcx, [e_mic]
    call fatal
.opened:
    lea rcx, [dev]
    call dev_show
    lea rcx, [rs]
    mov edx, [dev+DV_RATE]
    mov r8d, 16000
    call rs_init
    mov eax, [dev+DV_BUF]
    imul eax, [dev+DV_CH]
    lea rcx, [rax*4+4096]
    call mem_alloc
    mov [raw], rax
    mov ecx, [dev+DV_BUF]
    lea rcx, [rcx*4+4096]
    call mem_alloc
    mov [mono], rax
    say "  listening: speak, ctrl+c stops", 13, 10
    lea rcx, [dev]
    call dev_start
.loop:
    cmp dword [stop_flag], 0
    jne .stop
    lea rcx, [dev]
    call dev_wait
    test eax, eax
    jnz .loop
    lea rcx, [dev]
    mov rdx, [raw]
    mov r8d, [dev+DV_BUF]
    call dev_read
    test rax, rax
    jz .loop
    mov r12, rax                ; frames
    ; downmix to mono
    mov ecx, [dev+DV_CH]
    mov r8, [raw]
    mov r9, [mono]
    cvtsi2ss xmm2, ecx
    movss xmm3, [c_inv]
    divss xmm3, xmm2
    xor r10d, r10d
.fr:
    cmp r10, r12
    jae .frd
    xorps xmm0, xmm0
    xor r11d, r11d
.ch:
    addss xmm0, [r8]
    add r8, 4
    inc r11d
    cmp r11d, ecx
    jb .ch
    mulss xmm0, xmm3
    movss [r9+r10*4], xmm0
    inc r10
    jmp .fr
.frd:
    ; to 16 kHz, onto the line (a full line is kept, and a new one starts)
    lea rcx, [rs]
    mov rdx, r12
    call rs_maxout
    add rax, [npcm]
    cmp rax, [maxs]
    jbe .room
    call listen
    say 13, 10
    mov qword [npcm], 0
    mov qword [since], 0
.room:
    lea rcx, [rs]
    mov rdx, [mono]
    mov r8, r12
    mov r9, [npcm]
    shl r9, 2
    add r9, [pcm]
    call rs_push
    add [npcm], rax
    add [since], rax
    mov rax, [since]
    cmp rax, [every]
    jb .loop
    mov qword [since], 0
    call listen
    jmp .loop
.stop:
    lea rcx, [dev]
    call dev_stop
    call listen
    say 13, 10
    lea rcx, [dev]
    call dev_close
    xor ecx, ecx
    call ExitProcess

extern ExitProcess
