; batch DIR... [all=0] [frames=40000] [maxb=64] [fast=1] [maxsec=30] [fill=20]
;             [seed=1] [stats=datasets\melstats.bin] [tok=datasets\sptok.bin]
; the training data as the trainer will see it: opens DIR\*.shard (training ones
; only, unless all=1), keeps what ctc can train on, gets the mel normalization
; (worked out and saved the first time), plans an epoch of micro-batches of up to
; `frames` mel frames and fills a few: how full they are, and how fast
; aug=1 augments as training will (asr\augment's settings from the cfg), and
; wav=N writes the first N utterances of the first micro-batch, as it hears them,
; to scratch\aug\ with a line each on what was done to it
; uses: data\shard audio\flac audio\mel audio\wav audio\fft audio\resample asr\text asr\loader asr\augment tokenizer\pretok tokenizer\bpe tokenizer\tok
default rel
bits 64
%include "lib.inc"
%include "tokenizer/tok.inc"
%include "audio/audio.inc"
%include "asr/loader.inc"
%include "asr/augment.inc"

section .rdata
k_all    db "all", 0
k_frames db "frames", 0
k_maxb   db "maxb", 0
k_fast   db "fast", 0
k_maxsec db "maxsec", 0
k_fill   db "fill", 0
k_seed   db "seed", 0
k_stats  db "stats", 0
k_tok    db "tok", 0
d_stats  db "datasets\melstats.bin", 0
d_tok    db "datasets\sptok.bin", 0
e_tok    db "can't load the tokenizer", 0
e_none   db "no usable utterances", 0
usage    db "usage: batch DIR... [all=0] [frames=40000] [maxb=64] [fast=1] [maxsec=30] [fill=20]", 13, 10, 0
align 8
c_ms     dq 1000.0
c_hrs    dq 3600.0
c_pct    dq 100.0

section .bss
alignb 8
ld      resb LD_SIZE
hb      resb HB_SIZE
nfill   resq 1
real    resq 1                  ; mel frames that are speech, over the plan
padd    resq 1                  ; and that are paid for
text    resb 1 << 16
t0      resq 1
dirs    resq 64
ndirs   resq 1

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

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 56
    call lib_init
    ; the plain args (the directories) are noted before cfg_args cuts the others up
    mov ebx, 1
    xor r12d, r12d
.arg:
    cmp rbx, [argc]
    jae .args
    lea rax, [argv]
    mov rcx, [rax+rbx*8]
    mov r13, rcx
    call plain
    test eax, eax
    jz .na
    lea rax, [dirs]
    mov [rax+r12*8], r13
    inc r12
.na:
    inc rbx
    jmp .arg
.args:
    mov [ndirs], r12
    call cfg_args
    test r12, r12
    jnz .go
    lea rcx, [usage]
    call print_z
    xor ecx, ecx
    call ExitProcess
.go:
    xor ecx, ecx
    call pool_init
    lea rcx, [k_tok]
    lea rdx, [d_tok]
    call cfg_str
    mov rcx, rax
    call tok_load
    test eax, eax
    jnz .tok
    lea rcx, [e_tok]
    call fatal
.tok:
    lea rcx, [k_maxsec]
    mov edx, 30
    call cfg_int
    imul rax, 100
    mov [ld+LD_MAXT], rax
    imul rax, 160
    lea rcx, [rax+16000]        ; a second of room
    mov [maxs], rcx
    call ld_scratch
    lea rcx, [k_all]
    xor edx, edx
    call cfg_int
    mov [ld+LD_ALL], rax

    ; the shards
    xor ebx, ebx
.dir:
    cmp rbx, [ndirs]
    jae .index
    lea rax, [dirs]
    mov rdx, [rax+rbx*8]
    lea rcx, [ld]
    call ld_add
    inc rbx
    jmp .dir

.index:
    call time_now
    mov [t0], rax
    lea rcx, [ld]
    call ld_index
    mov rcx, [t0]
    call time_since
    movsd [rsp+32], xmm0
    say "  "
    mov rcx, [ld+LD_NSH]
    call print_dec
    say " shards: "
    mov rcx, [ld+LD_NU]
    call print_dec
    say " utterances kept, "
    movsd xmm0, [ld+LD_HOURS]
    mov edx, 1
    call print_fixed
    say " h (indexed in "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " s)", 13, 10, "  left out: "
    mov rcx, [ld+LD_DROP]
    call print_dec
    say " cleaning, "
    mov rcx, [ld+LD_DROP+8]
    call print_dec
    say " too long, "
    mov rcx, [ld+LD_DROP+16]
    call print_dec
    say " too short, "
    mov rcx, [ld+LD_DROP+24]
    call print_dec
    say " too dense for ctc", 13, 10
    cmp qword [ld+LD_NU], 0
    jne .norm
    lea rcx, [e_none]
    call fatal

.norm:
    call time_now
    mov [t0], rax
    lea rcx, [k_stats]
    lea rdx, [d_stats]
    call cfg_str
    lea rcx, [ld]
    mov rdx, rax
    call ld_norm
    mov ebx, eax
    mov rcx, [t0]
    call time_since
    movsd [rsp+32], xmm0
    say "  mel normalization: "
    test ebx, ebx
    jz .loaded
    say "worked out from 2000 utterances in "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " s and saved"
    jmp .means
.loaded:
    say "loaded"
.means:
    say ". bin 0: mean "
    cvtss2sd xmm0, [ld+LD_MEAN]
    mov edx, 2
    call print_fixed
    say " std "
    movss xmm1, [ld+LD_ISTD]
    mov eax, __?float32?__(1.0)
    movd xmm0, eax
    divss xmm0, xmm1
    cvtss2sd xmm0, xmm0
    mov edx, 2
    call print_fixed
    say ", bin 79: mean "
    cvtss2sd xmm0, [ld+LD_MEAN+79*4]
    mov edx, 2
    call print_fixed
    say " std "
    movss xmm1, [ld+LD_ISTD+79*4]
    mov eax, __?float32?__(1.0)
    movd xmm0, eax
    divss xmm0, xmm1
    cvtss2sd xmm0, xmm0
    mov edx, 2
    call print_fixed
    say 13, 10

    ; augmentation, if asked for
    lea rcx, [k_aug]
    xor edx, edx
    call cfg_int
    test rax, rax
    jz .plan
    lea rcx, [ag]
    call aug_config
    call time_now
    mov [t0], rax
    lea rcx, [ag]
    lea rdx, [d_noise]
    mov r8, [maxs]
    call aug_init
    lea rax, [ag]
    mov [ld+LD_AUG], rax
    mov rcx, [t0]
    call time_since
    movsd [rsp+32], xmm0
    say "  augmenting: noise pool of "
    movsd xmm0, [ag+AG_POOL]
    divsd xmm0, [c_hrs]
    mov edx, 1
    call print_fixed
    say " h decoded in "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " s", 13, 10

.plan:
    ; plan an epoch
    lea rcx, [k_frames]
    mov edx, 40000
    call cfg_int
    mov r12, rax
    lea rcx, [k_maxb]
    mov edx, 64
    call cfg_int
    mov r13, rax
    lea rcx, [k_fast]
    mov edx, 1
    call cfg_int
    mov r14, rax
    lea rcx, [k_seed]
    mov edx, 1
    call cfg_int
    mov r9, rax
    mov [rsp+32], r14
    lea rcx, [ld]
    mov rdx, r12
    mov r8, r13
    call ld_plan
    ; how full: speech frames over paid-for frames
    xor esi, esi
.full:
    cmp rsi, [ld+LD_NB]
    jae .fulld
    imul rdi, rsi, MP_SIZE
    add rdi, [ld+LD_B]
    mov eax, [rdi+MP_B]
    imul eax, [rdi+MP_T]
    add [padd], rax
    xor ecx, ecx
.fu:
    cmp ecx, [rdi+MP_B]
    jae .fn
    mov eax, [rdi+MP_START]
    add eax, ecx
    mov rdx, [ld+LD_ORDER]
    mov eax, [rdx+rax*4]
    imul rax, UR_SIZE
    add rax, [ld+LD_U]
    mov eax, [rax+UR_N]
    add eax, 159
    xor edx, edx
    mov r8d, 160
    div r8d
    add [real], rax
    inc ecx
    jmp .fu
.fn:
    inc rsi
    jmp .full
.fulld:
    say "  an epoch: "
    mov rcx, [ld+LD_NB]
    call print_dec
    say " micro-batches of up to "
    mov rcx, r12
    call print_dec
    say " mel frames, "
    cvtsi2sd xmm0, qword [real]
    cvtsi2sd xmm1, qword [padd]
    divsd xmm0, xmm1
    mulsd xmm0, [c_pct]
    mov edx, 1
    call print_fixed
    say "% of the frames are speech", 13, 10

    ; fill a few
    mov rcx, r12
    add rcx, 3072
    imul rcx, 80 * 4
    call mem_alloc
    mov [hb+HB_MEL], rax
    mov rcx, r13
    imul rcx, 1024 * 4
    call mem_alloc
    mov [hb+HB_LAB], rax
    mov rcx, r13
    shl rcx, 2
    call mem_alloc
    mov [hb+HB_TL], rax
    mov rcx, r13
    shl rcx, 2
    call mem_alloc
    mov [hb+HB_UL], rax
    lea rcx, [k_fill]
    mov edx, 20
    call cfg_int
    cmp rax, [ld+LD_NB]
    jbe .nf
    mov rax, [ld+LD_NB]
.nf:
    mov [nfill], rax
    xor esi, esi                ; frames filled
    call time_now
    mov [t0], rax
    xor ebx, ebx
.fill:
    cmp rbx, [nfill]
    jae .filled
    lea rcx, [ld]
    mov edx, ebx
    lea r8, [hb]
    call ld_fill
    mov eax, [hb+HB_B]
    imul eax, [hb+HB_T]
    add rsi, rax
    inc rbx
    jmp .fill
.filled:
    mov rcx, [t0]
    call time_since
    movsd [rsp+40], xmm0
    say "  filled "
    mov rcx, [nfill]
    call print_dec
    say " in "
    movsd xmm0, [rsp+40]
    mulsd xmm0, [c_ms]
    cvtsi2sd xmm1, qword [nfill]
    divsd xmm0, xmm1
    mov edx, 1
    call print_fixed
    say " ms each, "
    ; audio seconds (padded frames / 100) per wall second
    cvtsi2sd xmm0, rsi
    mov eax, 100
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    divsd xmm0, [rsp+40]
    mov edx, 0
    call print_fixed
    say "x real time", 13, 10

    ; the last one filled: its shape, and its first utterance
    say "  the last: "
    mov ecx, [hb+HB_B]
    call print_dec
    say " utterances x "
    mov ecx, [hb+HB_T]
    call print_dec
    say " frames, "
    mov ecx, [hb+HB_NLAB]
    call print_dec
    say " labels. first: "
    mov rax, [hb+HB_TL]
    mov ecx, [rax]
    call print_dec
    say " encoder frames, "
    mov rax, [hb+HB_UL]
    mov ecx, [rax]
    call print_dec
    say " tokens: ", 13, 10, "    "
    ; its labels back to text
    mov rdi, [hb+HB_UL]
    mov edi, [rdi]
    mov rsi, [hb+HB_LAB]
    lea r12, [text]
    xor ecx, ecx
.lt:
    cmp ecx, edi
    jae .ltd
    mov eax, [rsi+rcx*4]
    mov [r12+rcx*2], ax
    inc ecx
    jmp .lt
.ltd:
    lea rcx, [text]
    mov edx, edi
    lea r8, [text+4096]
    call tok_decode
    lea rcx, [text+4096]
    mov rdx, rax
    call print
    say 13, 10
    ; a peek at its mel: frame 50, bins 0, 20, 40, 79
    say "    normalized mel, frame 50, bins 0/20/40/79:"
    mov rsi, [hb+HB_MEL]
    add rsi, 50 * 80 * 4
    xor ebx, ebx
.pk:
    say " "
    lea rax, [peek]
    movzx eax, byte [rax+rbx]
    cvtss2sd xmm0, [rsi+rax*4]
    mov edx, 2
    call print_fixed
    inc ebx
    cmp ebx, 4
    jb .pk
    say 13, 10
    call wavs
    xor ecx, ecx
    call ExitProcess

; wav=N: the first micro-batch's first N utterances as heard, into scratch\aug
wavs:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    lea rcx, [k_wav]
    xor edx, edx
    call cfg_int
    mov r12, rax
    test r12, r12
    jz .done
    mov rax, [ld+LD_B]
    mov eax, [rax+MP_B]
    cmp r12, rax
    jbe .n
    mov r12, rax
.n:
    lea rcx, [path]             ; (make_path writes into its argument)
    lea rdx, [d_augdir]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [path]
    call make_path
    mov rcx, [maxs]
    imul rcx, 5                 ; 4 bytes a sample, room for 0.9x and a tail
    add rcx, 1 << 20
    call mem_alloc
    mov rsi, rax
    xor ebx, ebx
.w:
    cmp rbx, r12
    jae .done
    lea rcx, [ld]
    xor edx, edx
    mov r8d, ebx
    mov r9, rsi
    call ld_audio
    mov edi, edx                ; what got done
    mov [clip+CL_DATA], rsi
    mov [clip+CL_FRAMES], rax
    mov dword [clip+CL_RATE], 16000
    mov dword [clip+CL_CH], 1
    lea rcx, [path]
    lea rdx, [d_augdir]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_aug]
    call fmt_str
    mov rcx, rax
    mov rdx, rbx
    mov r8d, 2
    call fmt_dec0
    mov rcx, rax
    lea rdx, [s_wav]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [path]
    lea rdx, [clip]
    mov r8d, 16
    call wav_write
    say "  "
    lea rcx, [path]
    call print_z
    say ":"
    test edi, AI_SLOW
    jz .nf
    say " 0.9x"
.nf:
    test edi, AI_FAST
    jz .ns
    say " 1.1x"
.ns:
    test edi, AI_RIR
    jz .nr
    say " reverb"
.nr:
    test edi, AI_NOISE
    jz .nn
    say " "                     ; (say goes through rax: index after it)
    mov eax, edi
    shr eax, 8
    and eax, 255
    lea rcx, [knames]
    mov rcx, [rcx+rax*8]
    call print_z
    say " at "
    mov ecx, edi
    shr ecx, 16
    and ecx, 255
    call print_dec
    say " dB"
.nn:
    test edi, AI_GAIN
    jz .ng
    say " gain "
    mov eax, edi
    sar eax, 24
    movsxd rcx, eax
    call print_int
    say " dB"
.ng:
    test edi, edi
    jnz .eol
    say " as recorded"
.eol:
    say 13, 10
    inc rbx
    jmp .w
.done:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
peek    db 0, 20, 40, 79
k_aug   db "aug", 0
k_wav   db "wav", 0
d_noise db "datasets\shards\noise", 0
d_augdir db "scratch\aug\", 0
s_aug   db "aug_", 0
s_wav   db ".wav", 0
kn0     db "noise", 0
kn1     db "music", 0
kn2     db "babble", 0
kn3     db "street/indoor noise", 0
kn4     db "room tone", 0
align 8
knames  dq kn0, kn1, kn2, kn3, kn4
section .bss
alignb 8
ag      resb AG_SIZE
clip    resb CL_SIZE
path    resb 512
maxs    resq 1
extern ExitProcess
