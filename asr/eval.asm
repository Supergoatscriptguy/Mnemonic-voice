; eval PRESET [ckpt=PATH] [data=DIRS] [held=1] [show=4] [chunk=0]
; scores a checkpoint (the newest of the preset's run unless ckpt= says) on
; held-out speech: ctc loss per label, and word error rate by greedy decoding, as
; written (case and punctuation count) and normalized (lower case, no punctuation),
; for each shard and over all. data= and the model come from asr\<preset>.cfg like
; the trainer's; held=0 takes every shard in the directories, not just
; dev*/test*/valid*. show= prints that many reference/hypothesis pairs. chunk=N
; scores it streaming: attention in chunks of N encoder frames (80 ms each)
; uses: gpu\cuda gpu\kernels asr\model asr\ckpt asr\loader asr\augment asr\score asr\text data\shard audio\flac audio\mel audio\fft audio\resample tokenizer\pretok tokenizer\bpe tokenizer\tok
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "tokenizer/tok.inc"
%include "data/shard.inc"
%include "asr/model.inc"
%include "asr/loader.inc"
%include "asr/train.inc"

extern ctc_greedy, words, wdist

MAXDIRS equ 16
MAXSH   equ 4096
MAXWD   equ 4096                ; words a transcript can have here

section .rdata
s_asr    db "asr\", 0
s_cfg    db ".cfg", 0
s_ckdir  db "checkpoints\", 0
s_star   db "\step_*.ckpt", 0
s_tok    db "datasets\sptok.bin", 0
s_heavy  db "datasets\shards\heavy-small", 0
s_melete db "melete", 0
k_run    db "run", 0
k_ckpt   db "ckpt", 0
k_data   db "data", 0
k_held   db "held", 0
k_show   db "show", 0
k_chunk  db "chunk", 0
k_maxsec db "max_sec", 0
k_frames db "frames", 0
k_maxb   db "maxb", 0
k_fast   db "fast", 0
k_bf16   db "bf16", 0
e_preset db "no such preset (asr\<name>.cfg)", 0
e_tok    db "can't load the tokenizer (datasets\sptok.bin)", 0
e_ckpt   db "no checkpoint to score", 0
e_none   db "no utterances to score", 0
align 8
c_pct    dq 100.0

section .bss
alignb 8
run      resq 1
cfgpath  resb 512
dirs     resq MAXDIRS
ndirs    resq 1
maxt     resq 1
budget   resq 1
maxb     resq 1
fast     resq 1
bf16     resq 1
held     resq 1
show     resq 1
chunk    resq 1
flags    resq 1
va       resb LD_SIZE
hb       resb HB_SIZE
mb       resb MB_SIZE
d_mel    resq 1
d_lab    resq 1
d_tl     resq 1
d_ul     resq 1
d_loss   resq 1
d_best   resq 1
hloss    resq 1
hbest    resq 1
hdr      resb CK_SIZE
path     resb 512
pattern  resb 512
; per shard: utterances, labels, loss, word errors and reference words, as
; written and normalized
s_utt    resq MAXSH
s_lab    resq MAXSH
s_loss   resq MAXSH             ; f64
s_ex     resq MAXSH
s_exw    resq MAXSH
s_no     resq MAXSH
s_now    resq MAXSH
shown    resq 1
htok     resw 4096
rtok     resw 4096
htext    resb 1 << 15
rtext    resb 1 << 15
hw       resq MAXWD
rw       resq MAXWD

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

; "a, b,c" in rcx -> dirs[], cut in place
splitdirs:
    xor edx, edx
.skip:
    mov al, [rcx]
    test al, al
    jz .done
    cmp al, ' '
    je .sp
    cmp al, ','
    jne .word
.sp:
    inc rcx
    jmp .skip
.word:
    cmp rdx, MAXDIRS
    jae .done
    lea rax, [dirs]
    mov [rax+rdx*8], rcx
    inc rdx
.in:
    mov al, [rcx]
    test al, al
    jz .done
    cmp al, ','
    je .cut
    cmp al, ' '
    je .cut
    inc rcx
    jmp .in
.cut:
    mov byte [rcx], 0
    inc rcx
    jmp .skip
.done:
    mov [ndirs], rdx
    ret

%macro geti 3                   ; slot, key, default
    lea rcx, [%2]
    mov edx, %3
    call cfg_int
    mov [%1], rax
%endmacro

; rcx = loader, edx = micro-batch: fill it, upload it, mb set up for it
feed:
    push rbx
    sub rsp, 32
    lea r8, [hb]
    call ld_fill
    mov rcx, [d_mel]
    mov rdx, [hb+HB_MEL]
    mov r8d, [hb+HB_B]
    imul r8d, [hb+HB_T]
    imul r8, 80 * 4
    call gpu_up
    mov r8d, [hb+HB_B]
    imul r8d, [hb+HB_U]
    shl r8, 2
    test r8, r8
    jz .nolab
    mov rcx, [d_lab]
    mov rdx, [hb+HB_LAB]
    call gpu_up
.nolab:
    mov rcx, [d_tl]
    mov rdx, [hb+HB_TL]
    mov r8d, [hb+HB_B]
    shl r8, 2
    call gpu_up
    mov rcx, [d_ul]
    mov rdx, [hb+HB_UL]
    mov r8d, [hb+HB_B]
    shl r8, 2
    call gpu_up
    mov eax, [hb+HB_B]
    mov [mb+MB_B], eax
    mov eax, [hb+HB_T]
    mov [mb+MB_T], eax
    mov eax, [hb+HB_U]
    mov [mb+MB_UMAX], eax
    mov eax, [hb+HB_T]
    shr eax, 3
    mov [mb+MB_CS], eax         ; full context, one chunk
    mov dword [mb+MB_LEFT], 0
    mov rax, [chunk]            ; chunk=N: streaming, N encoder frames a chunk, all
    test rax, rax               ; the left context
    jz .full
    mov [mb+MB_CS], eax
    mov dword [mb+MB_LEFT], 0xffffffff
.full:
    add rsp, 32
    pop rbx
    ret

; rcx = rdi's micro-batch, ebx = utterance in it. scores it into its shard's sums,
; and shows it while show lasts
one:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov r12d, ebx               ; k
    ; which utterance, which shard
    mov eax, [rdi+MP_START]
    add eax, r12d
    mov rcx, [va+LD_ORDER]
    mov eax, [rcx+rax*4]
    imul rax, UR_SIZE
    add rax, [va+LD_U]
    movzx r13d, word [rax+UR_SH]
    lea rcx, [s_utt]
    inc qword [rcx+r13*8]
    mov rcx, [hb+HB_UL]
    mov eax, [rcx+r12*4]
    lea rcx, [s_lab]
    add [rcx+r13*8], rax
    mov rcx, [hloss]
    cvtss2sd xmm0, [rcx+r12*4]
    lea rcx, [s_loss]
    addsd xmm0, [rcx+r13*8]
    movsd [rcx+r13*8], xmm0
    ; hypothesis: greedy over its frames
    mov eax, [hb+HB_T]
    shr eax, 3
    imul eax, r12d
    mov rcx, [hbest]
    lea rcx, [rcx+rax*4]
    mov rax, [hb+HB_TL]
    mov edx, [rax+r12*4]
    mov r8, [mdl+MD_V]
    dec r8d
    lea r9, [htok]
    call ctc_greedy
    lea rcx, [htok]
    mov rdx, rax
    lea r8, [htext]
    call tok_decode
    mov rsi, rax                ; hypothesis bytes
    ; reference: its labels
    mov eax, [hb+HB_U]
    imul eax, r12d
    mov rcx, [hb+HB_LAB]
    lea rcx, [rcx+rax*4]
    mov rdx, [hb+HB_UL]
    mov edx, [rdx+r12*4]
    lea r8, [rtok]
    xor eax, eax
.cp:
    cmp eax, edx
    jae .cpd
    mov r9d, [rcx+rax*4]
    mov [r8+rax*2], r9w
    inc eax
    jmp .cp
.cpd:
    lea rcx, [rtok]
    lea r8, [rtext]
    call tok_decode
    mov rdi, rax                ; reference bytes
    ; as written, then normalized
    xor r14d, r14d
.mode:
    lea rcx, [rtext]
    mov rdx, rdi
    mov r8d, r14d
    lea r9, [rw]
    call words
    mov rbx, rax
    lea rcx, [htext]
    mov rdx, rsi
    mov r8d, r14d
    lea r9, [hw]
    call words
    lea rcx, [rw]
    mov rdx, rbx
    lea r8, [hw]
    mov r9, rax
    call wdist
    lea rcx, [s_ex]
    lea rdx, [s_exw]
    test r14d, r14d
    jz .acc
    lea rcx, [s_no]
    lea rdx, [s_now]
.acc:
    add [rcx+r13*8], rax
    add [rdx+r13*8], rbx
    inc r14d
    cmp r14d, 2
    jb .mode
    ; show it
    mov rax, [shown]
    cmp rax, [show]
    jae .out
    inc qword [shown]
    say "  REF "
    lea rcx, [rtext]
    mov rdx, rdi
    call print
    say 13, 10, "  HYP "
    lea rcx, [htext]
    mov rdx, rsi
    call print
    say 13, 10
.out:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; xmm0 = 100 * rcx / rdx, printed with 2 decimals then "%"
pct:
    sub rsp, 40
    cvtsi2sd xmm0, rcx
    cvtsi2sd xmm1, rdx
    divsd xmm0, xmm1
    mulsd xmm0, [c_pct]
    mov edx, 2
    call print_fixed
    say "%"
    add rsp, 40
    ret

; rbx = shard (or -1 for the total, sums in rax..: see below). one line of results
; r12 = utterances, r13 = labels, xmm6 is not touched: loss in [rsp+?] -> passed:
; rcx = utterances, rdx = labels, xmm0 = loss sum, r8 = ex errs, r9 = ex words,
; [rsp+40] = no errs, [rsp+48] = no words
line:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    mov rbx, r8
    mov rsi, r9
    mov r12, rdx
    movsd [rsp+32], xmm0
    mov [rsp+40], rcx
    call print_dec              ; utterances
    say " utterances: wer "
    mov rcx, [rsp+56+32+40]
    mov rdx, [rsp+56+32+48]
    call pct
    say " normalized, "
    mov rcx, rbx
    mov rdx, rsi
    call pct
    say " as written. ctc loss "
    movsd xmm0, [rsp+32]
    cvtsi2sd xmm1, r12
    divsd xmm0, xmm1
    mov edx, 3
    call print_fixed
    say " per label", 13, 10
    add rsp, 56
    pop r12
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
    sub rsp, 72
    call lib_init
    call preset
    lea rcx, [k_run]
    mov rdx, [run]
    call cfg_str
    mov [run], rax
    lea rcx, [k_data]
    lea rdx, [s_heavy]
    call cfg_str
    mov rcx, rax
    call splitdirs
    geti maxt, k_maxsec, 30
    imul rax, [maxt], 100
    mov [maxt], rax
    geti budget, k_frames, 40960
    geti maxb, k_maxb, 64
    geti fast, k_fast, 1
    geti bf16, k_bf16, 1
    geti held, k_held, 1
    geti show, k_show, 4
    geti chunk, k_chunk, 0
    mov rax, [fast]
    and eax, PL_FAST
    mov [flags], rax
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
    ; the data
    mov rcx, [maxt]
    imul rcx, 160
    add rcx, 16000
    call ld_scratch
    mov rax, [maxt]
    mov [va+LD_MAXT], rax
    mov eax, 2
    cmp qword [held], 0
    jne .h
    mov eax, 1
.h:
    mov [va+LD_ALL], rax
    xor ebx, ebx
.d:
    cmp rbx, [ndirs]
    jae .dd
    lea rax, [dirs]
    mov rdx, [rax+rbx*8]
    lea rcx, [va]
    call ld_add
    inc rbx
    jmp .d
.dd:
    lea rcx, [va]
    call ld_index
    cmp qword [va+LD_NU], 0
    jne .some
    lea rcx, [e_none]
    call fatal
.some:
    ; the model, sized as the trainer sizes it
    call model_config
    mov rax, [maxt]
    add rax, 127
    and rax, -128
    mov [mdl+MD_T], rax
    mov rcx, rax
    mov rax, [budget]
    add rax, rcx
    dec rax
    xor edx, edx
    div rcx
    mov [mdl+MD_B], rax
    mov rax, [mdl+MD_T]
    shr rax, 3
    mov [mdl+MD_UMAX], rax
    mov eax, [fast]
    mov [mdl_fast], eax
    mov ecx, 4
    cmp qword [bf16], 0
    je .es
    mov ecx, 2
.es:
    call model_setup
    ; the checkpoint: ckpt=, or the run's newest
    lea rcx, [k_ckpt]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .newest
    cmp byte [rax], 0
    je .newest
    mov rsi, rax
    lea rdi, [path]
.cpp:
    mov al, [rsi]
    mov [rdi], al
    inc rsi
    inc rdi
    test al, al
    jnz .cpp
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
    lea rsi, [hdr+CK_MEAN]
    lea rdi, [va+LD_MEAN]
    mov ecx, 160
    rep movsd
    say "  "
    lea rcx, [path]
    call print_z
    say ": step "
    mov rcx, [hdr+CK_STEP]
    call print_dec
    say ", "
    mov rcx, [va+LD_NU]
    call print_dec
    say " utterances to score ("
    movsd xmm0, [va+LD_HOURS]
    mov edx, 1
    call print_fixed
    say " h)", 13, 10

    ; buffers
    mov rbx, [budget]
    cmp rbx, [mdl+MD_T]
    jae .b
    mov rbx, [mdl+MD_T]
.b:
    imul rbx, 80 * 4
    mov rcx, rbx
    call gpu_host
    mov [hb+HB_MEL], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_mel], rax
    mov rbx, [maxb]
    imul rbx, [mdl+MD_UMAX]
    shl rbx, 2
    mov rcx, rbx
    call gpu_host
    mov [hb+HB_LAB], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_lab], rax
    mov rbx, [maxb]
    shl rbx, 2
    mov rcx, rbx
    call gpu_host
    mov [hb+HB_TL], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_tl], rax
    mov rcx, rbx
    call gpu_host
    mov [hb+HB_UL], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_ul], rax
    mov rcx, rbx
    call gpu_host
    mov [hloss], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_loss], rax
    mov rbx, [mdl+MD_M]
    shl rbx, 2
    mov rcx, rbx
    call gpu_host
    mov [hbest], rax
    mov rcx, rbx
    call gpu_alloc
    mov [d_best], rax
    mov rax, [d_mel]
    mov [mb+MB_MEL], rax
    mov rax, [d_lab]
    mov [mb+MB_LAB], rax
    mov rax, [d_tl]
    mov [mb+MB_TL], rax
    mov rax, [d_ul]
    mov [mb+MB_UL], rax
    mov rax, [d_loss]
    mov [mb+MB_LOSS], rax
    mov eax, __?float32?__(1.0)
    mov [mb+MB_SCALE], eax

    ; one plan, every micro-batch of it
    mov rax, [flags]
    mov [rsp+32], rax
    lea rcx, [va]
    mov rdx, [budget]
    mov r8, [maxb]
    mov r9d, 12345
    call ld_plan
    xor r12d, r12d
.mb:
    cmp r12, [va+LD_NB]
    jae .report
    lea rcx, [va]
    mov edx, r12d
    call feed
    lea rcx, [mb]
    mov edx, FW_LOSS
    call model_fwd
    mov rcx, [d_best]
    call model_argmax
    mov rcx, [hloss]
    mov rdx, [d_loss]
    mov r8d, [hb+HB_B]
    shl r8, 2
    call gpu_down
    mov rcx, [hbest]
    mov rdx, [d_best]
    mov r8d, [hb+HB_B]
    mov eax, [hb+HB_T]
    shr eax, 3
    imul r8d, eax
    shl r8, 2
    call gpu_down
    imul rdi, r12, MP_SIZE
    add rdi, [va+LD_B]
    xor ebx, ebx
.u:
    cmp ebx, [hb+HB_B]
    jae .mbn
    call one
    inc ebx
    jmp .u
.mbn:
    inc r12
    jmp .mb

.report:
    ; per shard, then all of them
    xor ebx, ebx
    xor r13d, r13d              ; totals: utterances
    xor r14d, r14d              ;   labels
    xorpd xmm0, xmm0
    movsd [rsp+56], xmm0        ;   loss
    lea rax, [tot]
    mov qword [rax], 0
    mov qword [rax+8], 0
    mov qword [rax+16], 0
    mov qword [rax+24], 0
.sh:
    cmp rbx, [va+LD_NSH]
    jae .all
    lea rax, [s_utt]
    cmp qword [rax+rbx*8], 0
    je .shn
    say "  "
    mov rax, [va+LD_NAMES]
    mov rcx, [rax+rbx*8]
    call print_z
    say ": "
    lea rax, [s_no]
    mov rcx, [rax+rbx*8]
    mov [rsp+32], rcx
    lea rax, [s_now]
    mov rcx, [rax+rbx*8]
    mov [rsp+40], rcx
    lea rax, [s_utt]
    mov rcx, [rax+rbx*8]
    add r13, rcx
    lea rax, [s_lab]
    mov rdx, [rax+rbx*8]
    add r14, rdx
    lea rax, [s_loss]
    movsd xmm0, [rax+rbx*8]
    movsd xmm1, [rsp+56]
    addsd xmm1, xmm0
    movsd [rsp+56], xmm1
    lea rax, [s_ex]
    mov r8, [rax+rbx*8]
    lea rax, [s_exw]
    mov r9, [rax+rbx*8]
    lea rax, [tot]
    add [rax], r8
    add [rax+8], r9
    mov r10, [rsp+32]
    add [rax+16], r10
    mov r10, [rsp+40]
    add [rax+24], r10
    call line
.shn:
    inc rbx
    jmp .sh
.all:
    say "  all: "
    lea rax, [tot]
    mov rcx, [rax+16]
    mov [rsp+32], rcx
    mov rcx, [rax+24]
    mov [rsp+40], rcx
    mov r8, [rax]
    mov r9, [rax+8]
    mov rcx, r13
    mov rdx, r14
    movsd xmm0, [rsp+56]
    call line
    xor ecx, ecx
    call ExitProcess

section .bss
tot     resq 4                  ; as written errors, words; normalized errors, words
extern ExitProcess
