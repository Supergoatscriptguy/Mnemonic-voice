; data augmentation for training (settings in asr\augment.inc):
;  - speed: 0.9x or 1.1x through the resampler (9 -> 10 and 11 -> 10 samples), one
;    for a whole micro-batch, picked by the planner so the padding can make room
;  - room reverb: a random impulse response (simulated rooms and measured ones),
;    lined up on its direct sound so the words don't move, the level kept
;  - noise: from a pool decoded at the start, a kind picked by weight, a random
;    stretch of it (looped when short), at a random snr. babble is 3-5 talkers
;  - spec augment on the normalized log-mel: bands of bins and stretches of frames
;    set to 0 (the mean)
; each utterance draws from its own rng (the loader seeds it from the plan and the
; utterance), so an epoch comes out the same every time
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"
%include "data/shard.inc"
%include "asr/augment.inc"

extern fft_init, fft_conv

MAXNS   equ 64                  ; noise and rir shards
RIRMAX  equ 16000               ; impulse responses are cut at 1 s
SEGMAX  equ 60 * 16000          ; a noise file gives the pool 60 s at most

; a pool segment
SG_START equ 0                  ; u64 first sample in the pool
SG_LEN   equ 8                  ; u32
SG_KIND  equ 12                 ; u32
SG_SIZE  equ 16
; a pick, while the pool is being built
PK_SH    equ 0                  ; u32 shard
PK_UT    equ 4                  ; u32 utterance
PK_START equ 8                  ; u32 first sample taken
PK_TAKE  equ 12                 ; u32 samples taken
PK_SIZE  equ 16

section .rdata
s_pat    db "\*.shard", 0
n_noise  db "musan-noise", 0
n_music  db "musan-music", 0
n_speech db "musan-speech", 0
n_demand db "demand", 0
n_room   db "but-room", 0
n_simrir db "simrir", 0
n_butrir db "but-rir", 0
align 8
names    dq n_noise, AK_NOISE, n_music, AK_MUSIC, n_speech, AK_BABBLE, n_demand, AK_STREET
         dq n_room, AK_ROOM, n_simrir, -1, n_butrir, -1, 0
align 4
kcum     dd 35, 50, 70, 85, 100 ; kinds by weight: cumulative, of 100
snrlo    dd 0, 5, 10, 0, 5      ; their snr ranges, dB
snrhi    dd 20, 20, 25, 20, 25
kshare   dd 35, 15, 20, 15, 15  ; their share of the pool, %
c_scale4 dd 3.0517578125e-05    ; 1/32768
align 8
c_db     dq 0.23025850929940458 ; ln(10) / 10
c_one    dq 1.0
c_rate   dq 16000.0
e_none   db "no noise shards in the noise directory", 0
e_flac   db "a noise or impulse response flac didn't decode", 0

section .bss
alignb 8
ag      resq 1
nsh     resq 1
sh      resb MAXNS * SR_SIZE
shkind  resq MAXNS
nrir    resq 1
rirs    resq 1                  ; (shard u32, utterance u32) of every impulse response
pool    resq 1                  ; int16
nseg    resq 1
segs    resq 1
kfirst  resq NKIND
kcount  resq NKIND
th_fd   resq 1                  ; per thread: an FD
th_a    resq 1                  ;   f32, th_n: the result
th_b    resq 1                  ;   f32, th_n: noise, reverb output
th_h    resq 1                  ;   f32 [RIRMAX]
th_z    resq 1                  ;   complex f32 [2^zlog]
th_rs   resq 1                  ;   an RS
th_n    resq 1
zlog    resq 1
picks   resq 1
pattern resb 1024
found   resb 1 << 16
rng     resb RNG_SIZE

section .text

; rcx = rng. rax = 64 random bits
rand:
    sub rsp, 40
    call rng_next
    add rsp, 40
    ret

; rcx = rng, edx = bound. eax = below it
below:
    push rbx
    sub rsp, 32
    mov ebx, edx
    call rng_next
    xor edx, edx
    div rbx
    mov eax, edx
    add rsp, 32
    pop rbx
    ret

; rcx = rng. xmm0 = uniform [0, 1)
unif:
    sub rsp, 40
    call rng_float
    add rsp, 40
    ret

; rcx = path. rax = its kind (-1 rir, -2 not ours): matched on the file's name
kindof:
    mov rdx, rcx
.end:
    mov al, [rcx]
    test al, al
    jz .ended
    inc rcx
    cmp al, 92
    je .cut
    cmp al, '/'
    jne .end
.cut:
    mov rdx, rcx
    jmp .end
.ended:
    lea r8, [names]
.n:
    mov r9, [r8]
    test r9, r9
    jz .no
    mov rcx, rdx
.c:
    mov al, [r9]
    test al, al
    jz .end2
    cmp al, [rcx]
    jne .next
    inc r9
    inc rcx
    jmp .c
.end2:
    cmp byte [rcx], '.'         ; the whole name, then ".shard"
    jne .next
    mov rax, [r8+8]
    ret
.next:
    add r8, 16
    jmp .n
.no:
    mov rax, -2
    ret

; rcx = shard #, edx = utterance, r8 = thread. its flac decoded into the thread's
; FD: rax = int16 samples (mem_free FD_PCM after), rdx = how many
dec1:
    push rbx
    sub rsp, 32
    imul rax, rcx, SR_SIZE
    lea r9, [sh]
    add rax, r9
    imul rcx, rdx, UT_SIZE
    add rcx, [rax+SR_INDEX]
    mov ebx, [rcx+UT_N]
    mov rdx, rcx
    mov rcx, [rdx+UT_FLAC]
    add rcx, [rax+SR_BASE]
    mov edx, [rdx+UT_FLEN]
    imul r8, FD_SIZE
    add r8, [th_fd]
    mov [rsp+40+8], r8          ; (home slot)
    call flac_decode
    test eax, eax
    jz .bad
    mov r8, [rsp+40+8]
    mov rax, [r8+FD_PCM]
    mov rdx, rbx
    add rsp, 32
    pop rbx
    ret
.bad:
    lea rcx, [e_flac]
    call fatal

; par_for over the picks: decode each, copy its stretch into the pool
fillpool:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
.p:
    cmp rsi, rdi
    jae .done
    imul rbx, rsi, PK_SIZE
    add rbx, [picks]
    mov ecx, [rbx+PK_SH]
    mov edx, [rbx+PK_UT]
    mov r8, r12
    call dec1
    imul rcx, rsi, SG_SIZE
    add rcx, [segs]
    mov rcx, [rcx+SG_START]
    shl rcx, 1
    add rcx, [pool]
    mov edx, [rbx+PK_START]
    lea rdx, [rax+rdx*2]
    mov r8d, [rbx+PK_TAKE]
.cp:
    test r8d, r8d
    jz .cpd
    mov r9w, [rdx]
    mov [rcx], r9w
    add rdx, 2
    add rcx, 2
    dec r8d
    jmp .cp
.cpd:
    mov rcx, r12
    imul rcx, FD_SIZE
    add rcx, [th_fd]
    mov rcx, [rcx+FD_PCM]
    call mem_free
    inc rsi
    jmp .p
.done:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = settings (AG_*), rdx = noise directory, r8 = the most samples an utterance
; has. opens the noise and impulse response shards, decodes the noise pool, makes
; every thread's scratch. after pool_init
global aug_init
aug_init:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov [ag], rcx
    mov rsi, rdx
    ; room for 0.9x and a reverb tail
    mov rax, r8
    imul rax, 10
    xor edx, edx
    mov ecx, 9
    div rcx
    add rax, 2 * RIRMAX + 64
    mov [th_n], rax
    add rax, RIRMAX
    xor ecx, ecx
.lg:
    mov edx, 1
    shl rdx, cl
    cmp rdx, rax
    jae .lgd
    inc ecx
    jmp .lg
.lgd:
    mov [zlog], rcx
    call fft_init
    ; the shards
    lea rcx, [pattern]
    mov rdx, rsi
    call fmt_str
    mov rcx, rax
    lea rdx, [s_pat]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [pattern]
    lea rdx, [found]
    mov r8d, 1 << 16
    call file_find
    mov rdi, rax
    mov rsi, rdx
.f:
    test rdi, rdi
    jz .opened
    mov rcx, [rsi]
    call kindof
    cmp rax, -2
    je .fn
    mov rbx, [nsh]
    cmp rbx, MAXNS
    jae .opened
    lea rcx, [shkind]
    mov [rcx+rbx*8], rax
    imul rdx, rbx, SR_SIZE
    lea rcx, [sh]
    add rdx, rcx
    mov rcx, [rsi]
    call shard_open
    test eax, eax
    jz .fn
    inc qword [nsh]
.fn:
    add rsi, 8
    dec rdi
    jmp .f
.opened:
    cmp qword [nsh], 0
    jne .some
    lea rcx, [e_none]
    call fatal
.some:
    ; every impulse response
    xor ebx, ebx
    xor r12d, r12d
.rc:
    cmp rbx, [nsh]
    jae .rcd
    lea rax, [shkind]
    cmp qword [rax+rbx*8], -1
    jne .rcn
    imul rax, rbx, SR_SIZE
    lea rcx, [sh]
    add r12, [rcx+rax+SR_COUNT]
.rcn:
    inc rbx
    jmp .rc
.rcd:
    mov [nrir], r12
    lea rcx, [r12*8+64]
    call mem_alloc
    mov [rirs], rax
    mov rdi, rax
    xor ebx, ebx
.rf:
    cmp rbx, [nsh]
    jae .rfd
    lea rax, [shkind]
    cmp qword [rax+rbx*8], -1
    jne .rfn
    imul rax, rbx, SR_SIZE
    lea rcx, [sh]
    mov rcx, [rcx+rax+SR_COUNT]
    xor edx, edx
.ru:
    cmp rdx, rcx
    jae .rfn
    mov [rdi], ebx
    mov [rdi+4], edx
    add rdi, 8
    inc rdx
    jmp .ru
.rfn:
    inc rbx
    jmp .rf
.rfd:
    ; the pool: each kind's files in a shuffled order until its share is full
    lea rcx, [rng]
    mov edx, 77
    call rng_seed
    mov ecx, 1 << 20
    imul rcx, PK_SIZE
    call mem_alloc
    mov [picks], rax
    mov ecx, 1 << 20
    imul rcx, SG_SIZE
    call mem_alloc
    mov [segs], rax
    xor r15d, r15d              ; pool samples so far
    xor r14d, r14d              ; kind
.kind:
    cmp r14, NKIND
    jae .kd
    mov rax, [nseg]
    lea rcx, [kfirst]
    mov [rcx+r14*8], rax
    ; this kind's budget in samples
    mov rax, [ag]
    movsd xmm0, [rax+AG_POOL]
    mulsd xmm0, [c_rate]
    lea rcx, [kshare]
    cvtsi2sd xmm1, dword [rcx+r14*4]
    mulsd xmm0, xmm1
    mov eax, 100
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    cvttsd2si r13, xmm0         ; budget
    xor ebx, ebx                ; shard
.ks:
    cmp rbx, [nsh]
    jae .kn
    lea rax, [shkind]
    cmp [rax+rbx*8], r14
    jne .ksn
    ; its utterances in a random order: start at a random one, step by a random
    ; odd number (coprime with... near enough: walk all of them once)
    imul rsi, rbx, SR_SIZE
    lea rax, [sh]
    add rsi, rax
    mov r12, [rsi+SR_COUNT]
    lea rcx, [rng]
    mov edx, r12d
    call below
    mov edi, eax                ; current
    xor eax, eax
    mov [rsp+32], rax           ; taken from this shard
.ku:
    cmp [rsp+32], r12
    jae .ksn
    test r13, r13
    jle .ksn
    mov rdx, [rsi+SR_INDEX]
    imul rcx, rdi, UT_SIZE
    mov eax, [rdx+rcx+UT_N]     ; its samples
    mov ecx, SEGMAX
    cmp eax, ecx
    cmova eax, ecx              ; take
    test eax, eax
    jz .kun
    mov r8, [nseg]
    imul r9, r8, PK_SIZE
    add r9, [picks]
    mov [r9+PK_SH], ebx
    mov [r9+PK_UT], edi
    mov [r9+PK_TAKE], eax
    mov dword [r9+PK_START], 0
    ; longer than SEGMAX: a random window of it
    mov rdx, [rsi+SR_INDEX]
    imul rcx, rdi, UT_SIZE
    mov ecx, [rdx+rcx+UT_N]
    sub ecx, eax
    jz .win
    mov [rsp+40], rax
    lea edx, [rcx+1]
    lea rcx, [rng]
    call below
    mov r8, [nseg]
    imul r9, r8, PK_SIZE
    add r9, [picks]
    mov [r9+PK_START], eax
    mov rax, [rsp+40]
.win:
    mov r8, [nseg]
    imul r9, r8, SG_SIZE
    add r9, [segs]
    mov [r9+SG_START], r15
    mov [r9+SG_LEN], eax
    mov [r9+SG_KIND], r14d
    add r15, rax
    sub r13, rax
    inc qword [nseg]
.kun:
    inc qword [rsp+32]
    inc rdi
    cmp rdi, r12
    jb .ku
    xor edi, edi
    jmp .ku
.ksn:
    inc rbx
    jmp .ks
.kn:
    mov rax, [nseg]
    lea rcx, [kfirst]
    sub rax, [rcx+r14*8]
    lea rcx, [kcount]
    mov [rcx+r14*8], rax
    inc r14
    jmp .kind
.kd:
    ; per thread scratch
    mov rcx, [nthreads]
    imul rcx, FD_SIZE
    call mem_alloc
    mov [th_fd], rax
    mov rcx, [nthreads]
    imul rcx, [th_n]
    shl rcx, 2
    mov rbx, rcx
    call mem_alloc
    mov [th_a], rax
    mov rcx, rbx
    call mem_alloc
    mov [th_b], rax
    mov rcx, [nthreads]
    imul rcx, RIRMAX * 4
    call mem_alloc
    mov [th_h], rax
    mov ecx, [zlog]
    mov eax, 8
    shl rax, cl
    imul rax, [nthreads]
    mov rcx, rax
    call mem_alloc
    mov [th_z], rax
    mov rcx, [nthreads]
    imul rcx, RS_SIZE
    call mem_alloc
    mov [th_rs], rax
    ; decode the pool
    lea rcx, [r15*2+64]
    call mem_alloc
    mov [pool], rax
    lea rcx, [fillpool]
    xor edx, edx
    mov r8, [nseg]
    mov r9d, 1
    call par_for
    mov rcx, [picks]
    call mem_free
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = speed code (0 none, 1 slow, 2 fast), rdx = samples. rax = samples after it.
; touches only rax, rcx, rdx (the planner keeps things in the rest)
global aug_out
aug_out:
    mov rax, rdx
    cmp ecx, 1
    jb .done
    je .slow
    mov ecx, 11
    jmp .r
.slow:
    mov ecx, 9
.r:
    imul rax, 10
    add rax, rcx
    dec rax
    xor edx, edx
    div rcx                     ; ceil(n 10 / M)
.done:
    ret

; ---- the pieces, on the thread's buffers. rbx = thread, r12 = rng, rsi = th_a
; (the samples), r13 = how many

; room reverb on [rsi, r13)
reverb:
    push rdi
    push r14
    push r15
    sub rsp, 48
    ; an impulse response, as f32
    mov rcx, r12
    mov edx, [nrir]
    call below
    mov rcx, [rirs]
    mov edx, [rcx+rax*8+4]
    mov ecx, [rcx+rax*8]
    mov r8, rbx
    call dec1
    mov r14, rdx
    cmp r14, RIRMAX
    jbe .m
    mov r14d, RIRMAX
.m:
    mov rdi, rbx
    imul rdi, RIRMAX * 4
    add rdi, [th_h]
    xor ecx, ecx
    xorps xmm2, xmm2            ; peak
    xor r15d, r15d              ; where
    movss xmm3, [c_scale4]
.h:
    cmp rcx, r14
    jae .hd
    movsx edx, word [rax+rcx*2]
    cvtsi2ss xmm0, edx
    mulss xmm0, xmm3
    movss [rdi+rcx*4], xmm0
    andps xmm0, [absmask]
    comiss xmm0, xmm2
    jbe .hn
    movaps xmm2, xmm0
    mov r15, rcx
.hn:
    inc rcx
    jmp .h
.hd:
    mov rcx, rbx
    imul rcx, FD_SIZE
    add rcx, [th_fd]
    mov rcx, [rcx+FD_PCM]
    call mem_free
    ; the energy before
    call energy
    movsd [rsp+40], xmm0
    ; zeros for the outputs past the end, then x * h, n + peak of it
    lea rax, [r13+r15]
    xor ecx, ecx
    mov rdx, r13
.z:
    cmp rdx, rax
    jae .zd
    mov [rsi+rdx*4], ecx
    inc rdx
    jmp .z
.zd:
    ; fft size: 2^k >= n + peak + m - 1
    lea rax, [r13+r15]
    add rax, r14
    xor ecx, ecx
.k:
    mov edx, 1
    shl rdx, cl
    cmp rdx, rax
    jae .kd
    inc ecx
    jmp .k
.kd:
    mov [rsp+32], rcx           ; (scratch slot for k, moved below)
    mov rax, rbx
    mov ecx, [zlog]
    mov edx, 8
    shl rdx, cl
    imul rax, rdx
    add rax, [th_z]
    mov r9, rbx
    imul r9, [th_n]
    shl r9, 2
    add r9, [th_b]              ; y
    sub rsp, 32
    mov [rsp+32], rax           ; scratch
    mov rax, [rsp+64]
    mov [rsp+40], rax           ; k
    mov [rsp+48], r9            ; y
    lea rdx, [r13+r15]
    mov rcx, rsi
    mov r8, rdi
    mov r9, r14
    call fft_conv
    add rsp, 32
    ; y[peak, peak + n) back over x, at the energy x had
    mov rdi, rbx
    imul rdi, [th_n]
    shl rdi, 2
    add rdi, [th_b]
    lea rdi, [rdi+r15*4]
    xor ecx, ecx
.cb:
    cmp rcx, r13
    jae .cbd
    mov eax, [rdi+rcx*4]
    mov [rsi+rcx*4], eax
    inc rcx
    jmp .cb
.cbd:
    call energy
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    jbe .out
    movsd xmm1, [rsp+40]
    divsd xmm1, xmm0
    sqrtsd xmm1, xmm1
    cvtsd2ss xmm1, xmm1
    xor ecx, ecx
.sc:
    cmp rcx, r13
    jae .out
    movss xmm0, [rsi+rcx*4]
    mulss xmm0, xmm1
    movss [rsi+rcx*4], xmm0
    inc rcx
    jmp .sc
.out:
    add rsp, 48
    pop r15
    pop r14
    pop rdi
    ret

; xmm0 = sum of squares of [rsi, r13), f64
energy:
    xorpd xmm0, xmm0
    xor ecx, ecx
.e:
    cmp rcx, r13
    jae .d
    cvtss2sd xmm1, [rsi+rcx*4]
    mulsd xmm1, xmm1
    addsd xmm0, xmm1
    inc rcx
    jmp .e
.d:
    ret

; noise of one kind into [rdi, r13): ecx = kind, edx = 1 to add (babble), 0 to set
; a random segment of it from a random place, looped
noise1:
    push r14
    push r15
    sub rsp, 56
    mov r14d, ecx
    mov r15d, edx
    lea rax, [kcount]
    mov edx, [rax+r14*8]
    mov rcx, r12
    call below
    lea rcx, [kfirst]
    add rax, [rcx+r14*8]
    imul rax, SG_SIZE
    add rax, [segs]
    mov r8, [rax+SG_START]
    shl r8, 1
    add r8, [pool]              ; its samples
    mov r9d, [rax+SG_LEN]
    mov [rsp+32], r8
    mov [rsp+40], r9
    mov rcx, r12
    mov edx, r9d
    call below
    mov r8, [rsp+32]
    mov r9, [rsp+40]
    mov edx, eax                ; where in it
    movss xmm2, [c_scale4]
    xor ecx, ecx
.n:
    cmp rcx, r13
    jae .done
    movsx eax, word [r8+rdx*2]
    cvtsi2ss xmm0, eax
    mulss xmm0, xmm2
    test r15d, r15d
    jz .set
    addss xmm0, [rdi+rcx*4]
.set:
    movss [rdi+rcx*4], xmm0
    inc edx
    cmp edx, r9d
    jb .nx
    xor edx, edx
.nx:
    inc rcx
    jmp .n
.done:
    add rsp, 56
    pop r15
    pop r14
    ret

; noise on [rsi, r13). eax = its info bits (0 if none got added)
addnoise:
    push rdi
    push r14
    push r15
    sub rsp, 48
    ; a kind by weight, one that has segments
    mov rcx, r12
    mov edx, 100
    call below
    xor r14d, r14d
    lea rcx, [kcum]
.w:
    cmp eax, [rcx+r14*4]
    jb .kw
    inc r14d
    jmp .w
.kw:
    xor r15d, r15d              ; tries
.try:
    lea rcx, [kcount]
    cmp qword [rcx+r14*8], 0
    jne .have
    inc r14d
    cmp r14d, NKIND
    jb .tn
    xor r14d, r14d
.tn:
    inc r15d
    cmp r15d, NKIND
    jb .try
    xor eax, eax
    jmp .out
.have:
    mov rdi, rbx
    imul rdi, [th_n]
    shl rdi, 2
    add rdi, [th_b]
    cmp r14d, AK_BABBLE
    jne .one
    ; 3 to 5 talkers
    mov rcx, r12
    mov edx, 3
    call below
    lea r15d, [rax+3]
    mov ecx, AK_BABBLE
    xor edx, edx
    call noise1
    dec r15d
.tk:
    mov ecx, AK_BABBLE
    mov edx, 1
    call noise1
    dec r15d
    jnz .tk
    jmp .mix
.one:
    mov ecx, r14d
    xor edx, edx
    call noise1
.mix:
    ; snr: the gain that puts noise snr dB under the speech
    call energy
    movsd [rsp+32], xmm0        ; speech
    push rsi
    mov rsi, rdi
    sub rsp, 8
    call energy
    add rsp, 8
    pop rsi
    movsd [rsp+40], xmm0        ; noise
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    jbe .none
    movsd xmm0, [rsp+32]
    comisd xmm0, xmm1
    jbe .none
    lea rax, [snrhi]
    mov edx, [rax+r14*4]
    lea rax, [snrlo]
    sub edx, [rax+r14*4]
    inc edx
    mov rcx, r12
    call below
    lea rcx, [snrlo]
    add eax, [rcx+r14*4]
    mov r15d, eax               ; snr, dB
    cvtsi2sd xmm0, eax
    mulsd xmm0, [c_db]
    call math_exp               ; 10^(snr/10)
    mulsd xmm0, [rsp+40]
    movsd xmm1, [rsp+32]
    divsd xmm1, xmm0
    sqrtsd xmm1, xmm1
    cvtsd2ss xmm1, xmm1
    xor ecx, ecx
.a:
    cmp rcx, r13
    jae .ad
    movss xmm0, [rdi+rcx*4]
    mulss xmm0, xmm1
    addss xmm0, [rsi+rcx*4]
    movss [rsi+rcx*4], xmm0
    inc rcx
    jmp .a
.ad:
    mov eax, r15d
    shl eax, 16
    mov ecx, r14d
    shl ecx, 8
    or eax, ecx
    or eax, AI_NOISE
    jmp .out
.none:
    xor eax, eax
.out:
    add rsp, 48
    pop r15
    pop r14
    pop rdi
    ret

; rcx = samples (f32), rdx = how many, r8d = speed code, r9 = thread, [rsp+40] = the
; utterance's rng. speed, reverb and noise as the settings and the rng say.
; rax = the result (the thread's buffer), rdx = its length, r8d = what got done (AI_*)
global aug_utt
aug_utt:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rbx, r9
    mov r12, [rsp+40+88]
    mov rdi, rcx
    mov r13, rdx
    mov r14d, r8d               ; becomes the info word
    mov rsi, rbx
    imul rsi, [th_n]
    shl rsi, 2
    add rsi, [th_a]
    test r14d, r14d
    jz .copy
    ; speed: 9 -> 10 (0.9x) or 11 -> 10 (1.1x)
    mov rcx, rbx
    imul rcx, RS_SIZE
    add rcx, [th_rs]
    mov [rsp+32], rcx
    mov edx, 9
    cmp r14d, 1
    je .in
    mov edx, 11
.in:
    mov r8d, 10
    call rs_init
    mov rcx, [rsp+32]
    mov rdx, rdi
    mov r8, r13
    mov r9, rsi
    call rs_push
    mov r13, rax
    mov rcx, [rsp+32]
    lea rdx, [rsi+r13*4]
    call rs_flush
    add r13, rax
    mov rcx, [rsp+32]
    call rs_free
    jmp .room
.copy:
    xor ecx, ecx
.cp:
    cmp rcx, r13
    jae .room
    mov eax, [rdi+rcx*4]
    mov [rsi+rcx*4], eax
    inc rcx
    jmp .cp
.room:
    mov rcx, r12
    call unif
    mov rax, [ag]
    comisd xmm0, [rax+AG_PRIR]
    jae .noise
    cmp qword [nrir], 0
    je .noise
    call reverb
    or r14d, AI_RIR
.noise:
    mov rcx, r12
    call unif
    mov rax, [ag]
    comisd xmm0, [rax+AG_PNOISE]
    jae .done
    call addnoise
    or r14d, eax
.done:
    ; gain, last: the whole thing as a quieter or louder mic would give it
    mov rax, [ag]
    xorpd xmm0, xmm0
    comisd xmm0, [rax+AG_PGAIN]
    jae .ret                    ; off: no draws, so the rest stays as it was
    mov rcx, r12
    call unif
    mov rax, [ag]
    comisd xmm0, [rax+AG_PGAIN]
    jae .ret
    mov rcx, r12
    call unif
    mov rax, [ag]
    movsd xmm1, [rax+AG_GHI]
    subsd xmm1, [rax+AG_GLO]
    mulsd xmm0, xmm1
    addsd xmm0, [rax+AG_GLO]
    cvtsd2si eax, xmm0
    shl eax, 24
    or r14d, eax
    or r14d, AI_GAIN
    mulsd xmm0, [c_db2ln]
    call math_exp
    cvtsd2ss xmm0, xmm0
    xor ecx, ecx
.g:
    cmp rcx, r13
    jae .ret
    movss xmm1, [rsi+rcx*4]
    mulss xmm1, xmm0
    movss [rsi+rcx*4], xmm1
    inc rcx
    jmp .g
.ret:
    mov rax, rsi
    mov rdx, r13
    mov r8d, r14d
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = log-mel rows (normalized, f32 [frames][80]), edx = frames, r8 = the
; utterance's rng. bands of bins and stretches of frames to 0
global aug_spec
aug_spec:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rsi, rcx
    mov r13d, edx
    mov r12, r8
    mov rbx, [ag]
    xor r14d, r14d
.f:
    cmp r14d, [rbx+AG_FMASK]
    jae .t
    mov rcx, r12
    mov edx, [rbx+AG_FW]
    inc edx
    call below
    mov edi, eax                ; width
    mov rcx, r12
    mov edx, MEL_N + 1
    sub edx, edi
    call below                  ; first bin
    xor ecx, ecx
.fr:
    cmp ecx, r13d
    jae .fn
    imul r8d, ecx, MEL_N * 4
    lea r8, [rsi+r8]
    xor edx, edx
.fb:
    cmp edx, edi
    jae .fbd
    lea r9d, [rax+rdx]
    mov dword [r8+r9*4], 0
    inc edx
    jmp .fb
.fbd:
    inc ecx
    jmp .fr
.fn:
    inc r14d
    jmp .f
.t:
    xor r14d, r14d
.tm:
    cmp r14d, [rbx+AG_TMASK]
    jae .done
    cvtsi2sd xmm0, r13d
    mulsd xmm0, [rbx+AG_TW]
    cvttsd2si edx, xmm0
    inc edx
    mov rcx, r12
    call below
    mov edi, eax                ; width
    mov rcx, r12
    mov edx, r13d
    sub edx, edi
    inc edx
    call below                  ; first frame
    imul eax, MEL_N * 4
    lea r8, [rsi+rax]
    imul ecx, edi, MEL_N
.tz:
    test ecx, ecx
    jz .tn
    mov dword [r8], 0
    add r8, 4
    dec ecx
    jmp .tz
.tn:
    inc r14d
    jmp .tm
.done:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
align 16
absmask dd 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff

section .rdata
k_pspeed db "p_speed", 0
k_prir   db "p_rir", 0
k_pnoise db "p_noise", 0
k_fmask  db "f_masks", 0
k_fw     db "f_width", 0
k_tmask  db "t_masks", 0
k_tw     db "t_width", 0
k_pool   db "noise_pool", 0
k_pgain  db "p_gain", 0
k_glo    db "gain_lo", 0
k_ghi    db "gain_hi", 0
align 8
c_db2ln  dq 0.11512925464970229 ; ln 10 / 20
section .text

%macro cfgf 3                   ; AG slot, key, default
    lea rcx, [%2]
    mov rax, __?float64?__(%3)
    movq xmm1, rax
    call cfg_float
    movsd [rbx+%1], xmm0
%endmacro
%macro cfgd 3                   ; AG u32 slot, key, default
    lea rcx, [%2]
    mov edx, %3
    call cfg_int
    mov [rbx+%1], eax
%endmacro

; rcx = AG. the settings from the cfg: p_speed (2/3: speed is 0.9x, 1x or 1.1x
; evenly), p_rir (0.3), p_noise (0.5), f_masks (2) of up to f_width (27) bins,
; t_masks (10) of up to t_width (0.05) of the frames, noise_pool (4 h, in seconds),
; p_gain (0) for a gain between gain_lo and gain_hi dB (-25, 5)
global aug_config
aug_config:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    cfgf AG_PSPEED, k_pspeed, 0.6666666666666666
    cfgf AG_PRIR, k_prir, 0.3
    cfgf AG_PNOISE, k_pnoise, 0.5
    cfgd AG_FMASK, k_fmask, 2
    cfgd AG_FW, k_fw, 27
    cfgd AG_TMASK, k_tmask, 10
    cfgf AG_TW, k_tw, 0.05
    cfgf AG_POOL, k_pool, 14400.0
    cfgf AG_PGAIN, k_pgain, 0.0
    cfgf AG_GLO, k_glo, -25.0
    cfgf AG_GHI, k_ghi, 5.0
    add rsp, 32
    pop rbx
    ret
