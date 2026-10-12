; train: the speech encoder's training loop.
;   bin\train melete              settings from asr\melete.cfg
;   bin\train melete lr=1e-3      key=value args override the file
; carries on from the newest checkpoint in checkpoints\<run>\ if there is one.
; ctrl+c finishes the step, saves and exits (code 2), a second one exits right away.
; the cpu fills the next micro-batch while the gpu runs this one: launches return
; at once, and the upload after the fill waits for the gpu to finish. "tok/s" and
; "tokens" on the bar are encoder frames (80 ms each)
; aug=1 augments the training data (asr\augment: speed, reverb, noise, spec augment)
; uses: gpu\cuda gpu\kernels asr\model asr\ckpt asr\loader asr\augment asr\score asr\text data\shard audio\flac audio\mel audio\fft audio\resample tokenizer\pretok tokenizer\bpe tokenizer\tok
default rel
bits 64
%include "lib.inc"
%include "gpu/cuda.inc"
%include "tokenizer/tok.inc"
%include "asr/model.inc"
%include "asr/loader.inc"
%include "asr/augment.inc"
%include "asr/train.inc"

MAXDIRS equ 16

section .rdata
s_asr    db "asr\", 0
s_cfg    db ".cfg", 0
s_ckdir  db "checkpoints\", 0
s_logdir db "logs\", 0
s_log    db ".log", 0
s_step   db "\step_", 0
s_ckext  db ".ckpt", 0
s_tmp    db ".tmp", 0
s_star   db "\step_*.ckpt", 0
s_stats  db "datasets\melstats.bin", 0
s_tok    db "datasets\sptok.bin", 0
s_heavy  db "datasets\shards\heavy-small", 0
s_melete db "melete", 0
k_run    db "run", 0                ; checkpoints\<run>, logs\<run>.log. the preset's name by default
k_data   db "data", 0
k_maxsec db "max_sec", 0
k_frames db "frames", 0
k_maxb   db "maxb", 0
k_accum  db "accum", 0
k_steps  db "steps", 0
k_lr     db "lr", 0
k_minlr  db "min_lr", 0
k_warmup db "warmup", 0
k_wd     db "wd", 0
k_clip   db "clip", 0
k_b2     db "b2", 0
k_fast   db "fast", 0
k_bf16   db "bf16", 0
k_seed   db "seed", 0
k_logev  db "log_every", 0
k_valev  db "val_every", 0
k_valb   db "val_batches", 0
k_saveev db "save_every", 0
k_keep   db "keep", 0
k_fresh  db "fresh", 0
k_aug    db "aug", 0
k_stream db "stream", 0                ; 1: half the micro-batches with chunked attention
s_noise  db "datasets\shards\noise", 0
m_aug    db "  augmenting: noise pool of ", 0
e_preset db "no such preset (asr\<name>.cfg)", 0
e_tok    db "can't load the tokenizer (datasets\sptok.bin)", 0
e_nodata db "no training utterances in the data directories", 0
m_resume db "  carrying on from ", 0
m_fresh  db "  fresh weights", 13, 10, 0
m_saved  db "  saved ", 0
m_val    db "  val: loss ", 0
m_per    db " per label over ", 0
m_utts   db " utterances", 13, 10, 0
m_stop   db "  stopping: saved, exit code 2", 13, 10, 0
align 8
c_hour   dq 3600.0
c_frame  dq 0.08                ; seconds per encoder frame
c_one    dq 1.0
c_half   dq 0.5
c_pi     dq 3.141592653589793
c_b1     dq 0.9
c_eps    dq 1e-8
c_nan    dq 0x7ff8000000000000

section .bss
alignb 8
; settings
run      resq 1
cfgpath  resb 512
dirs     resq MAXDIRS
ndirs    resq 1
maxt     resq 1                 ; mel frames an utterance may have
budget   resq 1                 ; mel frames a micro-batch may hold
maxb     resq 1
accum    resq 1
steps    resq 1
lr       resq 1                 ; f64
minlr    resq 1                 ; f64, fraction of lr where the cosine ends
warmup   resq 1
op       resb OP_SIZE
fast     resq 1
bf16     resq 1
seed     resq 1
logev    resq 1
valev    resq 1
valb     resq 1
saveev   resq 1
keep     resq 1
flags    resq 1                 ; for ld_plan
; state
tr       resb LD_SIZE           ; training data
va       resb LD_SIZE           ; held out
hb       resb HB_SIZE           ; a filled batch, pinned
mb       resb MB_SIZE
d_mel    resq 1
d_lab    resq 1
d_tl     resq 1
d_ul     resq 1
d_loss   resq 1                 ; accum * maxb losses
hloss    resq 1                 ; pinned host copy
hgn      resd 4
step     resq 1
epoch    resq 1
pos      resq 1                 ; next micro-batch in the epoch's plan
labels   resq 1                 ; in the step so far (for the log)
frames   resq 1                 ; encoder frames since the last log line
seen     resq 1                 ; encoder frames over the whole run
stepmb   resq 16                ; the step's micro-batches (accum of them)
ps       resb PS_SIZE
logf     resq 1
hdr      resb CK_SIZE
path     resb 512
tmp      resb 512
pattern  resb 512
msg      resb 4096
tlog     resq 1                 ; time of the last log line

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
    mov rdx, rcx
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
%macro getf 3                   ; slot, key, default (f64 constant)
    lea rcx, [%2]
    mov rax, __?float64?__(%3)
    movq xmm1, rax
    call cfg_float
    movsd [%1], xmm0
%endmacro

settings:
    sub rsp, 40
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
    geti accum, k_accum, 2
    cmp qword [accum], 16
    jbe .acc
    mov qword [accum], 16
.acc:
    geti steps, k_steps, 20000
    getf lr, k_lr, 1e-3
    getf minlr, k_minlr, 0.05
    geti warmup, k_warmup, 1000
    getf op+OP_WD, k_wd, 1e-3
    getf op+OP_CLIP, k_clip, 1.0
    getf op+OP_B2, k_b2, 0.98
    movsd xmm0, [c_b1]
    movsd [op+OP_B1], xmm0
    movsd xmm0, [c_eps]
    movsd [op+OP_EPS], xmm0
    geti fast, k_fast, 1
    geti bf16, k_bf16, 1
    geti seed, k_seed, 1
    geti logev, k_logev, 20
    geti valev, k_valev, 500
    geti valb, k_valb, 100
    geti saveev, k_saveev, 1000
    geti keep, k_keep, 3
    geti aug, k_aug, 0
    geti stream, k_stream, 0
    mov rax, [fast]
    and eax, PL_FAST
    mov [flags], rax
    add rsp, 40
    ret

; the loaders: training and held-out shards of every data directory, indexed,
; with one mel normalization
loaders:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rcx, [maxt]
    imul rcx, 160
    add rcx, 16000
    call ld_scratch
    mov rax, [maxt]
    mov [tr+LD_MAXT], rax
    mov [va+LD_MAXT], rax
    mov qword [va+LD_ALL], 2
    xor ebx, ebx
.d:
    cmp rbx, [ndirs]
    jae .dd
    lea rax, [dirs]
    mov rdx, [rax+rbx*8]
    lea rcx, [tr]
    call ld_add
    lea rax, [dirs]
    mov rdx, [rax+rbx*8]
    lea rcx, [va]
    call ld_add
    inc rbx
    jmp .d
.dd:
    lea rcx, [tr]
    call ld_index
    cmp qword [tr+LD_NU], 0
    jne .tr
    lea rcx, [e_nodata]
    call fatal
.tr:
    lea rcx, [va]
    call ld_index
    lea rcx, [tr]
    lea rdx, [s_stats]
    call ld_norm
    lea rsi, [tr+LD_MEAN]
    lea rdi, [va+LD_MEAN]
    mov ecx, 160
    rep movsd
    ; augmentation for the training set (never the held-out one)
    cmp qword [aug], 0
    je .noaug
    lea rcx, [ag]
    call aug_config
    lea rcx, [ag]
    lea rdx, [s_noise]
    mov r8, [maxt]
    imul r8, 160
    add r8, 16000
    call aug_init
    lea rax, [ag]
    mov [tr+LD_AUG], rax
    lea rcx, [m_aug]
    call print_z
    movsd xmm0, [ag+AG_POOL]
    divsd xmm0, [c_hour]
    mov edx, 1
    call print_fixed
    say " h", 13, 10
.noaug:
    say "  train: "
    mov rcx, [tr+LD_NU]
    call print_dec
    say " utterances, "
    movsd xmm0, [tr+LD_HOURS]
    mov edx, 1
    call print_fixed
    say " h. held out: "
    mov rcx, [va+LD_NU]
    call print_dec
    say " utterances, "
    movsd xmm0, [va+LD_HOURS]
    mov edx, 1
    call print_fixed
    say " h", 13, 10
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; the model, sized for the batches: frames an utterance rounded the way the plan
; rounds them, enough utterances for the frame budget, a label per encoder frame
setup:
    sub rsp, 40
    call model_config
    mov rax, [maxt]
    cmp qword [aug], 0
    je .t
    imul rax, 10                ; 0.9x makes them a ninth longer
    xor edx, edx
    mov ecx, 9
    div rcx
    inc rax
.t:
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
    call model_summary
    add rsp, 40
    ret

; pinned host and device buffers for one micro-batch, and the losses
buffers:
    push rbx
    sub rsp, 32
    ; mel: the budget, or one utterance at the most frames, whichever is more
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
    mov rbx, [maxb]
    imul rbx, [accum]
    shl rbx, 2
    mov rcx, rbx
    call gpu_alloc
    mov [d_loss], rax
    mov rcx, rbx
    call gpu_host
    mov [hloss], rax
    mov rax, [d_mel]
    mov [mb+MB_MEL], rax
    mov rax, [d_lab]
    mov [mb+MB_LAB], rax
    mov rax, [d_tl]
    mov [mb+MB_TL], rax
    mov rax, [d_ul]
    mov [mb+MB_UL], rax
    ; validation's greedy decoding: each encoder frame's best token
    mov rbx, [mdl+MD_M]
    shl rbx, 2
    mov rcx, rbx
    call gpu_alloc
    mov [d_best], rax
    mov rcx, rbx
    call gpu_host
    mov [hbest], rax
    add rsp, 32
    pop rbx
    ret

; checkpoints\<run>\ and the log
dirs_up:
    push rbx
    sub rsp, 32
    lea rcx, [path]
    lea rdx, [s_ckdir]
    call fmt_str
    mov rcx, rax
    mov rdx, [run]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [path]
    call make_path
    lea rcx, [path]
    lea rdx, [s_logdir]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [path]
    call make_path
    lea rcx, [path]
    lea rdx, [s_logdir]
    call fmt_str
    mov rcx, rax
    mov rdx, [run]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_log]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [path]
    call file_append
    mov [logf], rax
    add rsp, 32
    pop rbx
    ret

; rcx = buffer: checkpoints\<run>\step_<8 digits>.ckpt for step [step]. rax = end
ckname:
    push rbx
    sub rsp, 32
    lea rdx, [s_ckdir]
    call fmt_str
    mov rcx, rax
    mov rdx, [run]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_step]
    call fmt_str
    mov rcx, rax
    mov rdx, [step]
    mov r8d, 8
    call fmt_dec0
    mov rcx, rax
    lea rdx, [s_ckext]
    call fmt_str
    mov byte [rax], 0
    add rsp, 32
    pop rbx
    ret

; newest checkpoint, or fresh weights
resume:
    push rbx
    sub rsp, 32
    lea rcx, [pattern]          ; checkpoints\<run>\step_*.ckpt, for pruning too
    lea rdx, [s_ckdir]
    call fmt_str
    mov rcx, rax
    mov rdx, [run]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_star]
    call fmt_str
    mov byte [rax], 0
    cmp qword [fresh_], 0
    jne .fresh
    lea rcx, [pattern]
    lea rdx, [path]
    call ck_latest
    test eax, eax
    jz .fresh
    lea rcx, [path]
    lea rdx, [hdr]
    call ck_load
    mov rax, [hdr+CK_STEP]
    mov [step], rax
    mov rax, [hdr+CK_EPOCH]
    mov [epoch], rax
    mov rax, [hdr+CK_POS]
    mov [pos], rax
    mov rax, [hdr+CK_SEEN]
    mov [seen], rax
    movsd xmm0, [hdr+CK_ELAPSED]
    movsd [ps+PS_ELAPSED0], xmm0
    movsd xmm0, [hdr+CK_VLOSS]
    movsd [ps+PS_VLOSS], xmm0
    lea rcx, [m_resume]
    call print_z
    lea rcx, [path]
    call print_z
    say 13, 10
    jmp .done
.fresh:
    mov rcx, [seed]
    call model_init
    lea rcx, [m_fresh]
    call print_z
    mov rax, [c_nan]
    mov [ps+PS_VLOSS], rax
.done:
    add rsp, 32
    pop rbx
    ret

; this epoch's plan
plan:
    sub rsp, 40
    mov rax, [flags]
    mov [rsp+32], rax
    lea rcx, [tr]
    mov rdx, [budget]
    mov r8, [maxb]
    mov r9, [seed]
    imul r9, 1000003
    add r9, [epoch]
    call ld_plan
    add rsp, 40
    ret

; save where we are, keep the newest `keep`
save:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rax, [step]
    mov [hdr+CK_STEP], rax
    mov rax, [epoch]
    mov [hdr+CK_EPOCH], rax
    mov rax, [pos]
    mov [hdr+CK_POS], rax
    mov rax, [seen]
    mov [hdr+CK_SEEN], rax
    movsd xmm0, [ps+PS_ELAPSED]
    movsd [hdr+CK_ELAPSED], xmm0
    movsd xmm0, [ps+PS_VLOSS]
    movsd [hdr+CK_VLOSS], xmm0
    mov rax, [steps]
    mov [hdr+CK_TOTAL], rax
    lea rsi, [tr+LD_MEAN]
    lea rdi, [hdr+CK_MEAN]
    mov ecx, 160
    rep movsd
    lea rcx, [path]
    call ckname
    lea rcx, [tmp]
    call ckname
    mov rcx, rax
    lea rdx, [s_tmp]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [hdr]
    lea rdx, [path]
    lea r8, [tmp]
    call ck_save
    lea rcx, [pattern]
    mov rdx, [keep]
    call ck_prune
    lea rcx, [msg]
    lea rdx, [m_saved]
    call fmt_str
    mov rcx, rax
    lea rdx, [path]
    call fmt_str
    mov word [rax], 0x0a0d
    lea rdx, [msg]
    lea r8, [rax+2]
    sub r8, rdx
    lea rcx, [ps]
    call prog_msg
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = step index. xmm0 = its learning rate: linear warmup, then cosine down to
; min_lr * lr at the last step
lrat:
    sub rsp, 40
    cmp rcx, [warmup]
    jae .cos
    lea rax, [rcx+1]
    cvtsi2sd xmm0, rax
    cvtsi2sd xmm1, qword [warmup]
    divsd xmm0, xmm1
    mulsd xmm0, [lr]
    add rsp, 40
    ret
.cos:
    sub rcx, [warmup]
    cvtsi2sd xmm0, rcx
    mov rax, [steps]
    sub rax, [warmup]
    jle .min
    cvtsi2sd xmm1, rax
    divsd xmm0, xmm1
    minsd xmm0, [c_one]
    mulsd xmm0, [c_pi]
    call math_cos               ; 1 .. -1
    addsd xmm0, [c_one]
    mulsd xmm0, [c_half]        ; 1 .. 0
    movsd xmm1, [c_one]
    subsd xmm1, [minlr]
    mulsd xmm0, xmm1
    addsd xmm0, [minlr]
    mulsd xmm0, [lr]
    add rsp, 40
    ret
.min:
    movsd xmm0, [minlr]
    mulsd xmm0, [lr]
    add rsp, 40
    ret

; rcx = loader, edx = micro-batch: fill it, upload it, set mb up for it (the
; whole-step label scale and the loss slot are the caller's)
feed:
    push rbx
    push rsi
    sub rsp, 40
    mov rsi, rcx
    lea r8, [hb]
    call ld_fill
    mov rcx, [d_mel]
    mov rdx, [hb+HB_MEL]
    mov r8d, [hb+HB_B]
    imul r8d, [hb+HB_T]
    imul r8, 80 * 4
    call gpu_up
    mov rcx, [d_lab]
    mov rdx, [hb+HB_LAB]
    mov r8d, [hb+HB_B]
    imul r8d, [hb+HB_U]
    shl r8, 2
    test r8, r8
    jz .nolab
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
    ; full context: one chunk as long as the utterance. training with stream=1
    ; chunks half the micro-batches instead (2, 4, 8 or 16 encoder frames, all the
    ; left context), so the model learns to work streaming too
    mov eax, [hb+HB_T]
    shr eax, 3
    mov [mb+MB_CS], eax
    mov dword [mb+MB_LEFT], 0
    lea rax, [tr]
    cmp rsi, rax
    jne .ctx
    cmp qword [stream], 0
    je .ctx
    lea rcx, [srng]
    call rng_next
    test al, 1
    jz .ctx
    shr eax, 1
    and eax, 3
    mov ecx, eax
    mov eax, 2
    shl eax, cl
    mov [mb+MB_CS], eax
    mov dword [mb+MB_LEFT], 0xffffffff
.ctx:
    ; encoder frames of speech in it
    xor eax, eax
    xor ecx, ecx
    mov rdx, [hb+HB_TL]
.f:
    cmp ecx, [hb+HB_B]
    jae .fd
    add eax, [rdx+rcx*4]
    inc ecx
    jmp .f
.fd:
    add [frames], rax
    add rsp, 40
    pop rsi
    pop rbx
    ret

; one step: accum micro-batches, forward and backward, then adamw
onestep:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    ; a new epoch (and plan) only between steps, so a step's picks stay valid
    mov rax, [pos]
    add rax, [accum]
    cmp rax, [tr+LD_NB]
    jbe .same
    inc qword [epoch]
    mov qword [pos], 0
    call plan
.same:
    ; the step's micro-batches first, to count its labels (the loss is a mean over them)
    xor ebx, ebx
    xor edi, edi                ; labels
.pick:
    cmp rbx, [accum]
    jae .picked
    mov rdx, [pos]
    inc qword [pos]
    lea rax, [stepmb]
    mov [rax+rbx*8], rdx
    ; its labels, from the plan
    imul rcx, rdx, MP_SIZE
    add rcx, [tr+LD_B]
    mov r8d, [rcx+MP_START]
    mov r9d, [rcx+MP_B]
    xor r10d, r10d
.lab:
    cmp r10d, r9d
    jae .labd
    lea eax, [r8+r10]
    mov rdx, [tr+LD_ORDER]
    mov eax, [rdx+rax*4]
    imul rax, UR_SIZE
    add rax, [tr+LD_U]
    movzx eax, word [rax+UR_NT]
    add edi, eax
    inc r10d
    jmp .lab
.labd:
    inc rbx
    jmp .pick
.picked:
    mov [labels], rdi
    cvtsi2sd xmm0, rdi
    movsd xmm1, [c_one]
    divsd xmm1, xmm0
    cvtsd2ss xmm1, xmm1
    movd [mb+MB_SCALE], xmm1
    call model_zero
    xor ebx, ebx
.mb:
    cmp rbx, [accum]
    jae .step
    lea rax, [stepmb]
    mov edx, [rax+rbx*8]
    lea rcx, [tr]
    call feed
    mov rax, [maxb]
    imul rax, rbx
    shl rax, 2
    add rax, [d_loss]
    mov [mb+MB_LOSS], rax
    lea rcx, [mb]
    mov edx, FW_GRAD
    call model_fwd
    lea rcx, [mb]
    call model_bwd
    inc rbx
    jmp .mb
.step:
    mov rcx, [step]
    call lrat
    movsd [op+OP_LR], xmm0
    movsd [ps+PS_LR], xmm0
    lea rcx, [op]
    mov rdx, [step]
    inc rdx
    call model_step
    inc qword [step]
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; the last step's loss per label and gradient norm onto the bar, and a log line
report:
    push rbx
    sub rsp, 32
    mov rcx, [hloss]
    mov rdx, [d_loss]
    mov r8, [maxb]
    imul r8, [accum]
    shl r8, 2
    call gpu_down
    lea rcx, [hgn]
    mov rdx, [d_gn]
    mov r8d, 4
    call gpu_down
    ; the step's losses: each micro-batch's first B slots
    xorpd xmm0, xmm0
    xor ebx, ebx
.m:
    cmp rbx, [accum]
    jae .md
    lea rax, [stepmb]
    mov rax, [rax+rbx*8]
    imul rax, MP_SIZE
    add rax, [tr+LD_B]
    mov ecx, [rax+MP_B]
    mov rdx, [maxb]
    imul rdx, rbx
    shl rdx, 2
    add rdx, [hloss]
    xor eax, eax
.u:
    cmp eax, ecx
    jae .un
    cvtss2sd xmm1, [rdx+rax*4]
    addsd xmm0, xmm1
    inc eax
    jmp .u
.un:
    inc rbx
    jmp .m
.md:
    cvtsi2sd xmm1, qword [labels]
    divsd xmm0, xmm1
    movsd [ps+PS_LOSS], xmm0
    cvtss2sd xmm0, [hgn]
    sqrtsd xmm0, xmm0
    movsd [ps+PS_GNORM], xmm0
    ; encoder frames a second since the last report
    mov rcx, [tlog]
    call time_since
    cvtsi2sd xmm1, qword [frames]
    divsd xmm1, xmm0
    movsd [ps+PS_TOKS], xmm1
    mulsd xmm1, [mdl+MD_FLOPS]  ; training flops a frame, so tflops
    divsd xmm1, [c_tera]
    movsd [ps+PS_TFLOPS], xmm1
    call gpu_meminfo            ; rax free, rdx total
    mov [ps+PS_VRAMTOT], rdx
    sub rdx, rax
    mov [ps+PS_VRAM], rdx
    mov rax, [frames]
    add [seen], rax
    mov rax, [seen]
    mov [ps+PS_SEEN], rax
    mov qword [frames], 0
    call time_now
    mov [tlog], rax
    mov rax, [step]
    mov [ps+PS_STEP], rax
    lea rcx, [ps]
    mov rdx, [logf]
    call prog_line
    add rsp, 32
    pop rbx
    ret

; utterance ecx of the batch just run (losses and best tokens downloaded): greedy
; transcript against its labels, word errors into the vs_ sums. edx = 1 to show the
; pair above the bar
score1:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    mov r12d, ecx
    mov r13d, edx
    ; hypothesis
    mov eax, [hb+HB_T]
    shr eax, 3
    imul eax, r12d
    mov rcx, [hbest]
    lea rcx, [rcx+rax*4]
    mov rax, [hb+HB_TL]
    mov edx, [rax+r12*4]
    mov r8, [mdl+MD_V]
    dec r8d
    lea r9, [vtok]
    call ctc_greedy
    lea rcx, [vtok]
    mov rdx, rax
    lea r8, [htext]
    call tok_decode
    mov rsi, rax
    ; reference: its labels
    mov eax, [hb+HB_U]
    imul eax, r12d
    mov rcx, [hb+HB_LAB]
    lea rcx, [rcx+rax*4]
    mov rdx, [hb+HB_UL]
    mov edx, [rdx+r12*4]
    lea r8, [vtok]
    xor eax, eax
.cp:
    cmp eax, edx
    jae .cpd
    mov r9d, [rcx+rax*4]
    mov [r8+rax*2], r9w
    inc eax
    jmp .cp
.cpd:
    lea rcx, [vtok]
    lea r8, [rtext]
    call tok_decode
    mov rdi, rax
    ; as written, then normalized
    xor ebx, ebx
.mode:
    lea rcx, [rtext]
    mov rdx, rdi
    mov r8d, ebx
    lea r9, [rwd]
    call words
    mov [rsp+24+32], rax        ; (home slot: reference words)
    lea rcx, [htext]
    mov rdx, rsi
    mov r8d, ebx
    lea r9, [hwd]
    call words
    lea rcx, [rwd]
    mov rdx, [rsp+24+32]
    lea r8, [hwd]
    mov r9, rax
    call wdist
    mov rdx, [rsp+24+32]
    test ebx, ebx
    jnz .norm
    add [vs_ex], rax
    add [vs_exw], rdx
    jmp .mn
.norm:
    add [vs_no], rax
    add [vs_now], rdx
.mn:
    inc ebx
    cmp ebx, 2
    jb .mode
    test r13d, r13d
    jz .out
    ; "    REF ..." and "    HYP ..." above the bar, the first 200 bytes of each
    lea rcx, [msg]
    lea rdx, [m_ref]
    call fmt_str
    mov rcx, rax
    lea rdx, [rtext]
    mov r8, rdi
    call clip200
    mov rcx, rax
    lea rdx, [m_hyp]
    call fmt_str
    mov rcx, rax
    lea rdx, [htext]
    mov r8, rsi
    call clip200
    mov word [rax], 0x0a0d
    lea rdx, [msg]
    lea r8, [rax+2]
    sub r8, rdx
    lea rcx, [ps]
    call prog_msg
.out:
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = out, rdx = text, r8 = its length: copies at most 200 bytes, ending on a
; character boundary. rax = out's end
clip200:
    cmp r8, 200
    jbe .n
    mov r8d, 200
.back:
    mov al, [rdx+r8]
    and al, 0xc0
    cmp al, 0x80
    jne .n
    dec r8
    jmp .back
.n:
    xor eax, eax
.c:
    cmp rax, r8
    jae .d
    mov r9b, [rdx+rax]
    mov [rcx+rax], r9b
    inc rax
    jmp .c
.d:
    add rax, rcx
    ret

; the held-out set: loss per label, word error rate by greedy decoding (normalized,
; and as written), over up to val_batches of its micro-batches, and the first
; three clips' transcripts under their references (the plan is fixed, so they're
; the same clips every time)
validate:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    xorpd xmm0, xmm0
    movsd [vs_loss], xmm0
    xor eax, eax
    mov [vs_lab], rax
    mov [vs_utt], rax
    mov [vs_ex], rax
    mov [vs_exw], rax
    mov [vs_no], rax
    mov [vs_now], rax
    mov eax, __?float32?__(1.0)
    mov [mb+MB_SCALE], eax
    mov rax, [d_loss]
    mov [mb+MB_LOSS], rax
    xor ebx, ebx
.b:
    cmp rbx, [va+LD_NB]
    jae .done
    cmp rbx, [valb]
    jae .done
    lea rcx, [va]
    mov edx, ebx
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
    mov r8d, [hb+HB_T]
    shr r8d, 3
    imul r8d, [hb+HB_B]
    shl r8, 2
    call gpu_down
    mov ecx, [hb+HB_B]
    add [vs_utt], rcx
    mov eax, [hb+HB_NLAB]
    add [vs_lab], rax
    mov rdx, [hloss]
    movsd xmm0, [vs_loss]
    xor eax, eax
.u:
    cmp eax, ecx
    jae .un
    cvtss2sd xmm1, [rdx+rax*4]
    addsd xmm0, xmm1
    inc eax
    jmp .u
.un:
    movsd [vs_loss], xmm0
    xor r12d, r12d
.s:
    cmp r12d, [hb+HB_B]
    jae .sn
    xor edx, edx
    test rbx, rbx
    jnz .sc
    cmp r12d, 3
    setb dl
.sc:
    mov ecx, r12d
    call score1
    inc r12d
    jmp .s
.sn:
    inc rbx
    jmp .b
.done:
    mov qword [frames], 0       ; val frames aren't training throughput
    movsd xmm0, [vs_loss]
    cvtsi2sd xmm1, qword [vs_lab]
    divsd xmm0, xmm1
    movsd [ps+PS_VLOSS], xmm0
    lea rcx, [msg]
    lea rdx, [m_val]
    call fmt_str
    mov rcx, rax
    movsd xmm1, [ps+PS_VLOSS]
    mov r8d, 4
    call fmt_fixed
    mov rcx, rax
    lea rdx, [m_wer]
    call fmt_str
    mov rcx, rax
    mov rdx, [vs_no]
    mov r8, [vs_now]
    call fmt_pct
    mov rcx, rax
    lea rdx, [m_aswr]
    call fmt_str
    mov rcx, rax
    mov rdx, [vs_ex]
    mov r8, [vs_exw]
    call fmt_pct
    mov rcx, rax
    lea rdx, [m_over]
    call fmt_str
    mov rcx, rax
    mov rdx, [vs_utt]
    call fmt_dec
    mov rcx, rax
    lea rdx, [m_utts]
    call fmt_str
    lea rdx, [msg]
    mov r8, rax
    sub r8, rdx
    lea rcx, [ps]
    call prog_msg
    ; and into the log
    mov rcx, [logf]
    lea rdx, [msg]
    mov r8, rax
    call file_write
    call time_now
    mov [tlog], rax
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = out, rdx = errors, r8 = words: "12.34%". rax = out's end
fmt_pct:
    sub rsp, 40
    cvtsi2sd xmm1, rdx
    cvtsi2sd xmm2, r8
    divsd xmm1, xmm2
    mulsd xmm1, [c_hundred]
    mov r8d, 2
    call fmt_fixed
    mov byte [rax], '%'
    inc rax
    add rsp, 40
    ret

section .rdata
m_wer    db ", wer ", 0
m_aswr   db " (", 0
m_over   db " as written) over ", 0
m_ref    db "    REF ", 0
m_hyp    db 13, 10, "    HYP ", 0
align 8
c_hundred dq 100.0
c_tera   dq 1e12
section .bss
alignb 8
vs_loss  resq 1
vs_lab   resq 1
vs_utt   resq 1
vs_ex    resq 1
vs_exw   resq 1
vs_no    resq 1
vs_now   resq 1
d_best   resq 1
hbest    resq 1
vtok     resw 8192
rwd      resq 4096
hwd      resq 4096
htext    resb 1 << 15
rtext    resb 1 << 15
section .text

global start
start:
    sub rsp, 40
    call lib_init
    call preset
    geti fresh_, k_fresh, 0
    call settings
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
    call loaders
    call setup
    call buffers
    call dirs_up
    call resume
    lea rcx, [srng]
    mov rdx, [seed]
    imul rdx, rdx, 7919
    add rdx, [step]
    call rng_seed
    call plan
    ; the held-out set: one fixed plan
    mov rax, [flags]
    mov [rsp+32], rax
    lea rcx, [va]
    mov rdx, [budget]
    mov r8, [maxb]
    mov r9d, 12345
    call ld_plan
    ; prof=1: time one step's kernels and stop
    lea rcx, [k_prof]
    xor edx, edx
    call cfg_int
    test rax, rax
    jz .bar
    call profile
    xor ecx, ecx
    call ExitProcess
.bar:
    ; the bar
    mov rax, [step]
    mov [ps+PS_STEP], rax
    mov rax, [steps]
    mov [ps+PS_TOTAL], rax
    lea rcx, [ps]
    call prog_begin
    call time_now
    mov [tlog], rax
.loop:
    mov rax, [step]
    cmp rax, [steps]
    jae .finish
    cmp dword [stop_flag], 0
    jne .stop
    call onestep
    ; report, validate, save as their counts come up
    mov rax, [step]
    xor edx, edx
    div qword [logev]
    test rdx, rdx
    jnz .nr
    call report
.nr:
    mov rax, [step]
    mov [ps+PS_STEP], rax
    lea rcx, [ps]
    xor edx, edx
    call prog_draw
    mov rax, [step]
    xor edx, edx
    div qword [valev]
    test rdx, rdx
    jnz .nv
    call validate
.nv:
    mov rax, [step]
    xor edx, edx
    div qword [saveev]
    test rdx, rdx
    jnz .loop
    call save
    jmp .loop
.finish:
    call validate
    call save
    lea rcx, [ps]
    call prog_end
    xor ecx, ecx
    call ExitProcess
.stop:
    call save
    lea rcx, [ps]
    call prog_end
    lea rcx, [m_stop]
    call print_z
    mov ecx, 2
    call ExitProcess

; prof=1: two steps to warm up, then one with every launch timed. prints the
; kernels by total time (name, launches, ms, share), and the step's wall time
profile:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    call onestep
    call onestep
    call gpu_sync
    call time_now
    mov [pr_t0], rax
    call gpu_prof_begin
    call onestep
    xor ecx, ecx
    call gpu_prof_end
    mov rsi, rax                ; launches
    mov rcx, [pr_t0]
    call time_since
    movsd [pr_wall], xmm0
    ; sum by kernel
    xor ebx, ebx
    mov qword [pr_n], 0
.l:
    cmp rbx, rsi
    jae .sorted
    lea rax, [prof_fn]
    mov rcx, [rax+rbx*8]
    lea rax, [prof_ms]
    cvtss2sd xmm0, [rax+rbx*4]
    xor edx, edx
.f:
    cmp rdx, [pr_n]
    jae .new
    lea rax, [pr_fn]
    cmp [rax+rdx*8], rcx
    je .add
    inc rdx
    jmp .f
.new:
    cmp rdx, 128
    jae .ln
    lea rax, [pr_fn]
    mov [rax+rdx*8], rcx
    lea rax, [pr_ms]
    mov qword [rax+rdx*8], 0
    lea rax, [pr_cnt]
    mov qword [rax+rdx*8], 0
    inc qword [pr_n]
.add:
    lea rax, [pr_ms]
    addsd xmm0, [rax+rdx*8]
    movsd [rax+rdx*8], xmm0
    lea rax, [pr_cnt]
    inc qword [rax+rdx*8]
.ln:
    inc rbx
    jmp .l
.sorted:
    ; the total of the kernel times
    xorpd xmm0, xmm0
    xor ecx, ecx
.t:
    cmp rcx, [pr_n]
    jae .td
    lea rax, [pr_ms]
    addsd xmm0, [rax+rcx*8]
    inc rcx
    jmp .t
.td:
    movsd [pr_tot], xmm0
    say "  one step: "
    mov rcx, rsi
    call print_dec
    say " launches, "
    movsd xmm0, [pr_tot]
    mov edx, 1
    call print_fixed
    say " ms of kernels, "
    movsd xmm0, [pr_wall]
    mulsd xmm0, [c_thou]
    mov edx, 1
    call print_fixed
    say " ms wall", 13, 10
    ; biggest first: pick the largest left each time
.pick:
    xorpd xmm1, xmm1
    mov rdi, -1
    xor ecx, ecx
.p:
    cmp rcx, [pr_n]
    jae .pd
    lea rax, [pr_ms]
    movsd xmm0, [rax+rcx*8]
    comisd xmm0, xmm1
    jbe .pn
    movapd xmm1, xmm0
    mov rdi, rcx
.pn:
    inc rcx
    jmp .p
.pd:
    cmp rdi, -1
    je .done
    movsd [rsp+32], xmm1
    say "    "
    lea rax, [pr_fn]
    mov rcx, [rax+rdi*8]
    call gpu_fname
    mov rcx, rax
    call print_z
    say " x"
    lea rax, [pr_cnt]
    mov rcx, [rax+rdi*8]
    call print_dec
    say ": "
    movsd xmm0, [rsp+32]
    mov edx, 2
    call print_fixed
    say " ms ("
    movsd xmm0, [rsp+32]
    divsd xmm0, [pr_tot]
    mulsd xmm0, [c_pct]
    mov edx, 1
    call print_fixed
    say "%)", 13, 10
    lea rax, [pr_ms]
    mov qword [rax+rdi*8], 0
    jmp .pick
.done:
    ; the 10 slowest single launches (a big one after a small kernel is the gpu
    ; waiting on the cpu: the time to the next launch counts as the kernel's)
    say "  the slowest launches:"
    xor r12d, r12d
.top:
    cmp r12d, 10
    jae .topd
    xorps xmm1, xmm1
    mov rdi, -1
    xor ecx, ecx
.tl:
    cmp rcx, rsi
    jae .tld
    lea rax, [prof_ms]
    movss xmm0, [rax+rcx*4]
    comiss xmm0, xmm1
    jbe .tln
    movaps xmm1, xmm0
    mov rdi, rcx
.tln:
    inc rcx
    jmp .tl
.tld:
    cmp rdi, -1
    je .topd
    cvtss2sd xmm1, xmm1
    movsd [rsp+32], xmm1
    lea rax, [prof_ms]
    mov dword [rax+rdi*4], 0
    say " "
    lea rax, [prof_fn]
    mov rcx, [rax+rdi*8]
    call gpu_fname
    mov rcx, rax
    call print_z
    say "#"
    mov rcx, rdi
    call print_dec
    say "="
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    inc r12d
    jmp .top
.topd:
    say 13, 10
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
k_prof  db "prof", 0
align 8
c_thou  dq 1000.0
c_pct   dq 100.0
section .bss
fresh_  resq 1
aug     resq 1
stream  resq 1
srng    resb RNG_SIZE               ; picks the chunked micro-batches
ag      resb AG_SIZE
pr_t0   resq 1
pr_wall resq 1
pr_tot  resq 1
pr_n    resq 1
pr_fn   resq 128
pr_ms   resq 128
pr_cnt  resq 128
extern ExitProcess
extern ctc_greedy, words, wdist
