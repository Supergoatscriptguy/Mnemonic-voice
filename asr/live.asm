; live PRESET [ckpt=PATH] [secs=30] [every=400] [chunk=0] [agc=1] [level=-21] [wav=FILE] [beam=0 lm=FILE]
; dictation, live: the default mic through the speech model while you talk. every
; `every` ms the utterance so far goes through the encoder (all of it, again: the
; simple way, before the encoder keeps state between chunks) and its greedy
; transcript is redrawn in place. after secs seconds the line is kept and a new one
; starts. chunk=N attends in chunks of N encoder frames, as streaming would.
; agc: the speech is brought to `level` dBFS first (the training audio's speech sits
; around -21, a headset mic can be 20 dB under). wav=FILE: that file instead of the
; mic, through the same path. beam=W lm=FILE [alpha=0.15 beta=1 topk=8]: the beam
; search with the word model instead of greedy. ctrl+c stops
; uses: gpu\cuda gpu\kernels asr\model asr\ckpt asr\score audio\dev audio\resample audio\mel audio\wav audio\level tokenizer\pretok tokenizer\bpe tokenizer\tok asr\beam asr\lm
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "audio/audio.inc"
%include "tokenizer/tok.inc"
%include "asr/model.inc"
%include "asr/train.inc"
%include "asr/lm.inc"

extern ctc_greedy
extern beam_decode, bm_width, bm_topk, bm_alpha, bm_beta, bm_lm

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
k_agc    db "agc", 0
k_level  db "level", 0
k_beam   db "beam", 0
k_topk   db "topk", 0
k_lm     db "lm", 0
k_alpha  db "alpha", 0
k_beta   db "beta", 0
e_lm     db "can't load that language model", 0
k_wav    db "wav", 0
e_wav    db "can't read that wav", 0
e_preset db "no such preset (asr\<name>.cfg)", 0
e_tok    db "can't load the tokenizer", 0
e_ckpt   db "no checkpoint to listen with", 0
e_mic    db "the mic doesn't give float32", 0
align 8
c_gmin   dq -30.0
c_gmax   dq 30.0
c_db2ln  dq 0.11512925464970229 ; ln 10 / 20
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
spcm     resq 1                 ; it with the gain on
agc      resq 1
target   resq 1                 ; dBFS
gain     resq 1                 ; f64 dB, the last one used
whole    resq 1                 ; show all of the transcript (a file), not its end
beamw    resq 1
hlp      resq 1                 ; log probabilities for the beam search
lmst     resb LM_SIZE
fmean    resd 4096
fsort    resd 4096
bsum     resd 8
clip     resb CL_SIZE
clip16   resb CL_SIZE
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

; rcx = normalized log-mel [frames][MEL_N], rdx = frames. for the loudest 30% of the
; frames (the speech, about), each 10-bin band's mean: 0 is where the training
; audio's average sits, in its std. shows a mic that cuts the highs or colours them
bands:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rcx
    mov rbx, rdx
    cmp rbx, 4096
    jbe .n
    mov ebx, 4096
.n:
    test rbx, rbx
    jz .out
    ; each frame's loudness (its bins summed), and a copy to sort
    xor ecx, ecx
.m:
    cmp rcx, rbx
    jae .md
    imul rdx, rcx, MEL_N * 4
    add rdx, rsi
    xorps xmm0, xmm0
    xor eax, eax
.mb:
    addss xmm0, [rdx+rax*4]
    inc eax
    cmp eax, MEL_N
    jb .mb
    lea rax, [fmean]
    movss [rax+rcx*4], xmm0
    lea rax, [fsort]
    movss [rax+rcx*4], xmm0
    inc rcx
    jmp .m
.md:
    lea r8, [fsort]
    mov ecx, 1
.is:
    cmp rcx, rbx
    jae .isd
    movss xmm0, [r8+rcx*4]
    mov rdx, rcx
.isl:
    test rdx, rdx
    jz .ins
    movss xmm1, [r8+rdx*4-4]
    comiss xmm1, xmm0
    jbe .ins
    movss [r8+rdx*4], xmm1
    dec rdx
    jmp .isl
.ins:
    movss [r8+rdx*4], xmm0
    inc rcx
    jmp .is
.isd:
    imul rax, rbx, 7
    xor edx, edx
    mov ecx, 10
    div rcx
    movss xmm5, [r8+rax*4]      ; the 70th percentile
    lea r10, [bsum]
    xor eax, eax
.bz:
    mov dword [r10+rax*4], 0
    inc eax
    cmp eax, 8
    jb .bz
    xor ecx, ecx
    xor r12d, r12d
.f:
    cmp rcx, rbx
    jae .fd
    lea rax, [fmean]
    movss xmm0, [rax+rcx*4]
    comiss xmm0, xmm5
    jb .fn
    inc r12
    imul rdx, rcx, MEL_N * 4
    add rdx, rsi
    xor eax, eax
.fb:
    imul r9d, eax, 205          ; /10 for bins under 80
    shr r9d, 11
    movss xmm0, [rdx+rax*4]
    addss xmm0, [r10+r9*4]
    movss [r10+r9*4], xmm0
    inc eax
    cmp eax, MEL_N
    jb .fb
.fn:
    inc rcx
    jmp .f
.fd:
    say "  speech frames, mel bands low to high, in training std:"
    imul rax, r12, 10
    cvtsi2ss xmm0, rax
    movss [rsp+32], xmm0
    xor edi, edi
.p:
    say " "
    lea rax, [bsum]
    movss xmm0, [rax+rdi*4]
    divss xmm0, [rsp+32]
    cvtss2sd xmm0, xmm0
    mov edx, 2
    call print_fixed
    inc edi
    cmp edi, 8
    jb .p
    say 13, 10
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the line so far through the model; its transcript redrawn
listen:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    ; the speech to the training audio's level
    mov rsi, [pcm]
    cmp qword [agc], 0
    je .mel
    mov rcx, [pcm]
    mov rdx, [npcm]
    mov r8d, 16000
    mov r9d, 90
    call level_db
    cvtsi2sd xmm1, dword [target]
    subsd xmm1, xmm0
    maxsd xmm1, [c_gmin]
    minsd xmm1, [c_gmax]
    movsd [gain], xmm1
    mulsd xmm1, [c_db2ln]
    movapd xmm0, xmm1
    call math_exp
    cvtsd2ss xmm0, xmm0
    mov rcx, [npcm]
    mov rdx, [pcm]
    mov r8, [spcm]
    xor eax, eax
.g:
    cmp rax, rcx
    jae .gd
    movss xmm1, [rdx+rax*4]
    mulss xmm1, xmm0
    movss [r8+rax*4], xmm1
    inc rax
    jmp .g
.gd:
    mov rsi, [spcm]
.mel:
    ; log-mel, normalized with the checkpoint's statistics, zeros to T (a multiple
    ; of 1024 mel frames, so B*T3 is a multiple of 128 for the fast matmuls)
    lea rcx, [mel]
    mov rdx, rsi
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
    cmp qword [whole], 0
    je .nb
    mov rcx, rsi
    mov rdx, rbx
    call bands
.nb:
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
    cmp qword [beamw], 0
    jne .beam
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
    jmp .decode
.beam:
    call model_lsm
    mov rcx, [hlp]
    mov rdx, rax
    mov r8d, [tl]
    imul r8, VOCAB * 4
    call gpu_down
    mov rcx, [hlp]
    mov edx, [tl]
    lea r8, [toks]
    call beam_decode
.decode:
    lea rcx, [toks]
    mov rdx, rax
    lea r8, [text]
    call tok_decode
    ; redraw: the last 110 bytes that fit, from a character boundary
    mov rsi, rax
    lea rdi, [text]
    cmp qword [whole], 0
    jne .show
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
    cmp qword [agc], 0
    je .text
    xorpd xmm0, xmm0
    comisd xmm0, [gain]
    ja .neg
    say "+"
.neg:
    movsd xmm0, [gain]
    xor edx, edx
    call print_fixed
    say " dB | "
.text:
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
    lea rcx, [k_agc]
    mov edx, 1
    call cfg_int
    mov [agc], rax
    lea rcx, [k_level]
    mov edx, -21
    call cfg_int
    mov [target], rax
    ; beam=W: the beam search, with lm= if given
    lea rcx, [k_beam]
    xor edx, edx
    call cfg_int
    mov ecx, 32
    cmp rax, rcx
    cmova rax, rcx
    mov [beamw], rax
    test rax, rax
    jz .nobeam
    mov [bm_width], rax
    lea rcx, [k_topk]
    mov edx, 8
    call cfg_int
    mov ecx, 15
    cmp rax, rcx
    cmova rax, rcx
    mov [bm_topk], rax
    lea rcx, [k_alpha]
    mov rax, __?float64?__(0.15)
    movq xmm1, rax
    call cfg_float
    movsd [bm_alpha], xmm0
    mov qword [bm_beta], 0
    lea rcx, [k_lm]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .nobeam
    cmp byte [rax], 0
    je .nobeam
    mov rcx, rax
    lea rdx, [lmst]
    call lm_load
    test eax, eax
    jnz .lmok
    lea rcx, [e_lm]
    call fatal
.lmok:
    lea rax, [lmst]
    mov [bm_lm], rax
    lea rcx, [k_beta]
    mov rax, __?float64?__(1.0)
    movq xmm1, rax
    call cfg_float
    movsd [bm_beta], xmm0
.nobeam:
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
    mov rbx, rcx
    call mem_alloc
    mov [pcm], rax
    mov rcx, rbx
    call mem_alloc
    mov [spcm], rax
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
    mov rcx, [mdl+MD_T3]
    imul rcx, VOCAB * 4
    call gpu_host
    mov [hlp], rax

    ; wav=: a file instead, once
    lea rcx, [k_wav]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .mic
    cmp byte [rax], 0
    je .mic
    mov rcx, rax
    lea rdx, [clip]
    call wav_read
    test eax, eax
    jnz .wread
    lea rcx, [e_wav]
    call fatal
.wread:
    lea rcx, [clip]
    call clip_mono
    lea rcx, [clip]
    mov edx, 16000
    lea r8, [clip16]
    call clip_resample
    mov rcx, [clip16+CL_FRAMES]
    cmp rcx, [maxs]
    jbe .fits
    mov rcx, [maxs]
.fits:
    mov [npcm], rcx
    mov rsi, [clip16+CL_DATA]
    mov rdi, [pcm]
    rep movsd
    mov qword [whole], 1
    call listen
    say 13, 10
    call con_restore
    xor ecx, ecx
    call ExitProcess

.mic:
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
