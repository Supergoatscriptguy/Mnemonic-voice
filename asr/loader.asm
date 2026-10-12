; the speech data loader. ld_add opens a directory's shards, ld_index cleans and
; tokenizes every transcript and keeps what ctc can train on, ld_norm gets the mel
; normalization, ld_plan shuffles and cuts micro-batches of similar length under a
; frame budget, ld_fill decodes, computes log-mel and fills one on every thread.
; the tokenizer has to be loaded (tok_load) and the thread pool up before ld_scratch
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"
%include "data/shard.inc"
%include "tokenizer/tok.inc"
%include "asr/loader.inc"

extern clean_text

MAXSH   equ 4096                ; shards a loader can hold
CLEANB  equ 1 << 16             ; cleaned text a thread can hold
CLEANS  equ CLEANB + 64            ; ...a thread apart: the tokenizer reads up to 8 bytes before a word
POOL    equ 4096                ; utterances sorted together when planning
NSTAT   equ 2000                ; utterances the mel statistics come from

section .rdata
s_pat   db "\*.shard", 0
e_flac  db "a flac in a shard didn't decode", 0
e_many  db "too many shards", 0
align 8
c_hours dq 57600000.0           ; samples in an hour
c_scale dq 3.0517578125e-05     ; 1/32768
c_one   dq 1.0
c_epsv  dq 1e-5

section .bss
alignb 8
mel     resb MEL_SIZE           ; the filterbank, shared (mel_run only reads it)
th_fd   resq 1                  ; per thread: an FD
th_pcm  resq 1                  ;   f32 samples, th_maxs each
th_maxs resq 1
th_mel  resq 1                  ;   log-mel for the statistics, th_maxs/160 + 1 frames
th_cln  resq 1                  ;   cleaned text, CLEANB
th_tok  resq 1                  ;   u16 tokens, CLEANB
th_ctx  resq 64                 ;   a tokenizer cache each
pattern resb 1024
found   resb 1 << 20
; ld_index's scratch: per utterance across all shards
ix_ld   resq 1
ix_base resq MAXSH + 1          ; first global index of each shard
ix_st   resq 1                  ; u32 per utterance: reason << 16 | tokens
ix_off  resq 1                  ; u32 per utterance: where its tokens go
; ld_norm's per thread sums: f64 [threads][160] (sum, then sum of squares)
nm_sum  resq 1
nm_ids  resq 1
; ld_fill's job
fl_ld   resq 1
fl_mp   resq 1
fl_hb   resq 1
rng     resb RNG_SIZE

section .text

; rcx = shard path. eax = 1 if it's a training shard (not dev*, test*, valid*)
training:
    mov rdx, rcx
.end:
    cmp byte [rcx], 0
    je .ended
    mov al, [rcx]
    inc rcx
    cmp al, 92
    je .cut
    cmp al, '/'
    jne .end
.cut:
    mov rdx, rcx
    jmp .end
.ended:
    mov eax, [rdx]
    cmp eax, 'dev.'
    je .no
    cmp eax, 'test'
    je .no
    cmp eax, 'vali'
    je .no
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; rcx = LD, rdx = directory. opens its shards (training ones only unless LD_ALL)
global ld_add
ld_add:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    cmp qword [rbx+LD_SH], 0
    jne .have
    mov ecx, MAXSH * SR_SIZE
    call mem_alloc
    mov [rbx+LD_SH], rax
.have:
    lea rcx, [pattern]
    mov rdx, rsi
    call fmt_str
    mov rcx, rax
    lea rdx, [s_pat]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [pattern]
    lea rdx, [found]
    mov r8d, 1 << 20
    call file_find
    mov rdi, rax
    mov rsi, rdx
.shard:
    test rdi, rdi
    jz .out
    cmp qword [rbx+LD_ALL], 0
    jne .open
    mov rcx, [rsi]
    call training
    test eax, eax
    jz .next
.open:
    mov r12, [rbx+LD_NSH]
    cmp r12, MAXSH
    jae .many
    imul rdx, r12, SR_SIZE
    add rdx, [rbx+LD_SH]
    mov rcx, [rsi]
    call shard_open
    test eax, eax
    jz .next
    inc qword [rbx+LD_NSH]
.next:
    add rsi, 8
    dec rdi
    jmp .shard
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
.many:
    lea rcx, [e_many]
    call fatal

; ecx = the most samples an utterance can have. per thread scratch, and the mel
; filterbank. after pool_init and tok_load
global ld_scratch
ld_scratch:
    push rbx
    push rsi
    sub rsp, 40
    mov ecx, ecx
    mov [th_maxs], rcx
    lea rcx, [mel]
    call mel_init
    mov rcx, [nthreads]
    imul rcx, FD_SIZE
    call mem_alloc
    mov [th_fd], rax
    mov rcx, [nthreads]
    imul rcx, [th_maxs]
    shl rcx, 2
    call mem_alloc
    mov [th_pcm], rax
    mov rax, [th_maxs]
    xor edx, edx
    mov ecx, MEL_HOP
    div rcx
    inc rax
    imul rax, MEL_N * 4
    imul rax, [nthreads]
    mov rcx, rax
    call mem_alloc
    mov [th_mel], rax
    mov rcx, [nthreads]
    imul rcx, CLEANS
    call mem_alloc
    add rax, 64                 ; each buffer has 64 readable bytes before it
    mov [th_cln], rax
    mov rcx, [nthreads]
    imul rcx, CLEANB * 2
    call mem_alloc
    mov [th_tok], rax
    xor ebx, ebx
.ctx:
    cmp rbx, [nthreads]
    jae .done
    call tok_cache_new
    lea rcx, [th_ctx]
    mov [rcx+rbx*8], rax
    inc rbx
    jmp .ctx
.done:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; ---- ld_index

; rcx = global utterance index. rax = its shard's SR, rdx = the UT entry, r8 = shard #
locate:
    xor r8d, r8d
    mov r9, [ix_ld]
    mov r9, [r9+LD_NSH]
.s:
    lea rax, [ix_base]
    cmp rcx, [rax+r8*8+8]
    jb .found
    inc r8
    cmp r8, r9
    jb .s
.found:
    lea rax, [ix_base]
    sub rcx, [rax+r8*8]         ; index in the shard
    mov rax, [ix_ld]
    imul r9, r8, SR_SIZE
    mov rax, [rax+LD_SH]
    add rax, r9
    imul rdx, rcx, UT_SIZE
    add rdx, [rax+SR_INDEX]
    ret

; rax = SR, rdx = UT, r9 = thread. eax = cleaned length in the thread's buffer (-1 =
; can't be used). keeps rdx
cleanit:
    push rbx
    sub rsp, 32
    mov rbx, rdx
    mov r8, r9
    imul r8, CLEANS
    add r8, [th_cln]
    mov rcx, [rbx+UT_TEXT]
    add rcx, [rax+SR_BASE]
    mov edx, [rbx+UT_TLEN]
    cmp edx, CLEANB / 3
    jae .no
    call clean_text
    test rax, rax
    jle .no
    jmp .out
.no:
    mov rax, -1
.out:
    mov rdx, rbx
    add rsp, 32
    pop rbx
    ret

; rcx = ctx (unused), rdx = start, r8 = end, r9 = thread. pass 1: per utterance,
; why it's left out (or 0) and how many tokens it has
idx1:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
.u:
    cmp rsi, rdi
    jae .done
    mov rcx, rsi
    call locate
    mov rbx, rax                ; SR
    mov r13, rdx                ; UT
    ; length: mel frames over the cap, or under 0.3 s
    mov eax, [r13+UT_N]
    add eax, MEL_HOP - 1
    xor edx, edx
    mov ecx, MEL_HOP
    div ecx
    mov rcx, [ix_ld]
    cmp rax, [rcx+LD_MAXT]
    mov r14d, 2 << 16
    ja .put
    mov r14d, 3 << 16
    cmp dword [r13+UT_N], 4800
    jb .put
    mov rax, rbx
    mov rdx, r13
    mov r9, r12
    call cleanit
    mov r14d, 1 << 16
    cmp rax, 0
    jle .put
    ; tokens
    mov r8, rax
    mov rcx, r12
    lea rax, [th_ctx]
    mov rcx, [rax+rcx*8]
    mov rdx, r12
    imul rdx, CLEANS
    add rdx, [th_cln]
    mov r9, r12
    imul r9, CLEANB * 2
    add r9, [th_tok]
    call tok_encode
    mov r14d, 4 << 16
    test rax, rax
    jz .put
    cmp rax, 0xffff
    jae .put
    ; ctc needs a frame per token, and one between equal neighbours, even at 1.1x
    mov rcx, r12
    imul rcx, CLEANB * 2
    add rcx, [th_tok]
    mov r8, rax                 ; need
    xor edx, edx
.rep:
    lea r9, [rdx+1]
    cmp r9, rax
    jae .counted
    mov r10w, [rcx+rdx*2]
    cmp r10w, [rcx+rdx*2+2]
    jne .nr
    inc r8
.nr:
    inc rdx
    jmp .rep
.counted:
    mov r9, rax
    mov eax, [r13+UT_N]
    imul rax, 10
    xor edx, edx
    mov ecx, 11
    div rcx                     ; samples at 1.1x
    add rax, MEL_HOP * 8 - 1
    xor edx, edx
    mov ecx, MEL_HOP * 8
    div rcx                     ; encoder frames, about
    cmp r8, rax
    ja .put                     ; r14d still says too dense
    mov r14d, r9d               ; kept: just the count
.put:
    mov rax, [ix_st]
    mov [rax+rsi*4], r14d
    inc rsi
    jmp .u
.done:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; pass 2: the kept ones' tokens into LD_TOK
idx2:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
.u:
    cmp rsi, rdi
    jae .done
    mov rax, [ix_st]
    mov eax, [rax+rsi*4]
    cmp eax, 0xffff
    ja .next
    mov rcx, rsi
    call locate
    mov r9, r12
    call cleanit
    mov r8, rax
    lea rax, [th_ctx]
    mov rcx, [rax+r12*8]
    mov rdx, r12
    imul rdx, CLEANS
    add rdx, [th_cln]
    mov r9, r12
    imul r9, CLEANB * 2
    add r9, [th_tok]
    mov r13, r9
    call tok_encode
    mov rcx, [ix_off]
    mov ecx, [rcx+rsi*4]
    mov rdx, [ix_ld]
    mov rdx, [rdx+LD_TOK]
    lea rdx, [rdx+rcx*2]
.copy:
    test rax, rax
    jz .next
    mov cx, [r13]
    mov [rdx], cx
    add r13, 2
    add rdx, 2
    dec rax
    jmp .copy
.next:
    inc rsi
    jmp .u
.done:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = LD (shards added, LD_MAXT set). builds the table of usable utterances and
; their tokens, counts what's left out and why
global ld_index
ld_index:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rbx, rcx
    mov [ix_ld], rcx
    ; global indices: shard s covers [base[s], base[s+1])
    xor eax, eax
    xor ecx, ecx
    lea rdx, [ix_base]
.b:
    mov [rdx+rcx*8], rax
    cmp rcx, [rbx+LD_NSH]
    jae .bd
    imul r8, rcx, SR_SIZE
    add r8, [rbx+LD_SH]
    add rax, [r8+SR_COUNT]
    inc rcx
    jmp .b
.bd:
    mov r12, rax                ; utterances in all
    lea rcx, [rax*4+64]
    call mem_alloc
    mov [ix_st], rax
    lea rcx, [r12*4+64]
    call mem_alloc
    mov [ix_off], rax
    lea rcx, [idx1]
    xor edx, edx
    mov r8, r12
    mov r9d, 256
    call par_for
    ; count, and the token offsets
    xor esi, esi                ; utterance
    xor edi, edi                ; kept
    xor r13d, r13d              ; tokens so far
.c:
    cmp rsi, r12
    jae .cd
    mov rax, [ix_st]
    mov eax, [rax+rsi*4]
    cmp eax, 0xffff
    jbe .keep
    shr eax, 16
    lea rcx, [rbx+LD_DROP-8]
    inc qword [rcx+rax*8]
    jmp .cn
.keep:
    mov rcx, [ix_off]
    mov [rcx+rsi*4], r13d
    add r13, rax
    inc rdi
.cn:
    inc rsi
    jmp .c
.cd:
    mov [rbx+LD_NU], rdi
    mov [rbx+LD_NTOK], r13
    lea rcx, [r13*2+64]
    call mem_alloc
    mov [rbx+LD_TOK], rax
    imul rcx, rdi, UR_SIZE
    add rcx, 64
    call mem_alloc
    mov [rbx+LD_U], rax
    lea rcx, [idx2]
    xor edx, edx
    mov r8, r12
    mov r9d, 256
    call par_for
    ; the records, in global order
    xor esi, esi
    mov rdi, [rbx+LD_U]
    xor r14d, r14d              ; samples kept
.r:
    cmp rsi, r12
    jae .rd
    mov rax, [ix_st]
    mov eax, [rax+rsi*4]
    cmp eax, 0xffff
    ja .rn
    mov [rdi+UR_NT], ax
    mov rcx, rsi
    call locate
    mov [rdi+UR_SH], r8w
    mov [rdi+UR_UT], ecx
    mov eax, [rdx+UT_N]
    mov [rdi+UR_N], eax
    add r14, rax
    mov rax, [ix_off]
    mov eax, [rax+rsi*4]
    mov [rdi+UR_TOK], eax
    add rdi, UR_SIZE
.rn:
    inc rsi
    jmp .r
.rd:
    cvtsi2sd xmm0, r14
    divsd xmm0, [c_hours]
    movsd [rbx+LD_HOURS], xmm0
    mov rcx, [ix_st]
    call mem_free
    mov rcx, [ix_off]
    call mem_free
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ---- decoding

; rcx = LD, rdx = UR, r8 = thread. the utterance as f32 in the thread's buffer.
; rax = samples
decode:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rdx
    mov r12, r8
    movzx eax, word [rsi+UR_SH]
    imul rax, SR_SIZE
    add rax, [rcx+LD_SH]
    mov ecx, [rsi+UR_UT]
    imul rcx, UT_SIZE
    add rcx, [rax+SR_INDEX]
    mov rbx, [rcx+UT_FLAC]
    add rbx, [rax+SR_BASE]
    mov edx, [rcx+UT_FLEN]
    mov rdi, r12
    imul rdi, FD_SIZE
    add rdi, [th_fd]
    mov rcx, rbx
    mov r8, rdi
    call flac_decode
    test eax, eax
    jz .bad
    ; int16 -> f32, at most th_maxs
    mov ecx, [rsi+UR_N]
    cmp rcx, [th_maxs]
    jbe .n
    mov rcx, [th_maxs]
.n:
    mov rbx, rcx
    mov r8, [rdi+FD_PCM]
    mov r9, r12
    imul r9, [th_maxs]
    shl r9, 2
    add r9, [th_pcm]
    movss xmm1, [c_scale4]
    xor eax, eax
.cv:
    cmp rax, rbx
    jae .cvd
    movsx edx, word [r8+rax*2]
    cvtsi2ss xmm0, edx
    mulss xmm0, xmm1
    movss [r9+rax*4], xmm0
    inc rax
    jmp .cv
.cvd:
    mov rcx, [rdi+FD_PCM]
    call mem_free
    mov rax, rbx
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
.bad:
    lea rcx, [e_flac]
    call fatal

section .rdata
align 4
c_scale4 dd 3.0517578125e-05
section .text

; ---- ld_norm

NMS     equ 168                 ; a thread's sums: [80] sum, [80] sum of squares, frames

; rcx = thread. rax = its mel buffer (th_maxs/160 + 1 frames)
melbuf:
    mov r8, rcx
    mov rax, [th_maxs]
    xor edx, edx
    mov ecx, MEL_HOP
    div rcx
    inc rax
    imul rax, MEL_N * 4
    imul rax, r8
    add rax, [th_mel]
    ret

; rcx = thread. rax = its f32 sample buffer
pcmbuf:
    mov rax, rcx
    imul rax, [th_maxs]
    shl rax, 2
    add rax, [th_pcm]
    ret

; rcx = LD, rdx = start, r8 = end, r9 = thread: log-mel of the chosen utterances,
; summed per bin into the thread's f64 sums
nm1:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov r14, rcx
    mov rsi, rdx
    mov rdi, r8
    mov r12, r9
    mov rcx, r12
    call melbuf
    mov r15, rax
    imul r13, r12, NMS * 8
    add r13, [nm_sum]
.u:
    cmp rsi, rdi
    jae .done
    mov rax, [nm_ids]
    mov eax, [rax+rsi*4]
    imul rdx, rax, UR_SIZE
    add rdx, [r14+LD_U]
    mov rcx, r14
    mov r8, r12
    call decode
    mov rbx, rax
    mov rcx, r12
    call pcmbuf
    mov rdx, rax
    lea rcx, [mel]
    mov r8, rbx
    mov r9, r15
    call mel_run
    cvtsi2sd xmm0, rax
    addsd xmm0, [r13+160*8]
    movsd [r13+160*8], xmm0
    mov rbx, r15
    mov rcx, rax
.f:
    test rcx, rcx
    jz .next
    xor edx, edx
.b:
    cvtss2sd xmm0, [rbx+rdx*4]
    movapd xmm1, xmm0
    mulsd xmm1, xmm0
    addsd xmm0, [r13+rdx*8]
    movsd [r13+rdx*8], xmm0
    addsd xmm1, [r13+rdx*8+MEL_N*8]
    movsd [r13+rdx*8+MEL_N*8], xmm1
    inc edx
    cmp edx, MEL_N
    jb .b
    add rbx, MEL_N * 4
    dec rcx
    jmp .f
.next:
    inc rsi
    jmp .u
.done:
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = LD (indexed), rdx = path of the statistics file. loads the per bin mean
; and 1/std from it, or works them out from up to NSTAT utterances spread over
; the set and saves them there. eax = 1 if it had to work them out
global ld_norm
ld_norm:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov rbx, rcx
    mov r13, rdx
    mov rcx, r13
    call file_read_all
    test rax, rax
    jz .work
    mov r12, rax
    cmp rdx, MEL_N * 8
    jne .notit
    lea rdi, [rbx+LD_MEAN]
    mov rsi, r12
    mov ecx, MEL_N * 2
    rep movsd
    mov rcx, r12
    call mem_free
    xor eax, eax
    jmp .out
.notit:
    mov rcx, r12
    call mem_free
.work:
    ; every k-th utterance
    mov ecx, NSTAT * 4
    call mem_alloc
    mov [nm_ids], rax
    mov rax, [rbx+LD_NU]
    xor edx, edx
    mov ecx, NSTAT
    div rcx
    test rax, rax
    jnz .k
    mov eax, 1
.k:
    mov r8, [nm_ids]
    xor ecx, ecx
    xor edx, edx
.pick:
    cmp ecx, NSTAT
    jae .picked
    cmp rdx, [rbx+LD_NU]
    jae .picked
    mov [r8+rcx*4], edx
    inc ecx
    add rdx, rax
    jmp .pick
.picked:
    mov r12, rcx
    mov rcx, [nthreads]
    imul rcx, NMS * 8
    mov rsi, rcx
    call mem_alloc
    mov [nm_sum], rax
    mov rdi, rax
    mov rcx, rsi
    xor eax, eax
    rep stosb
    lea rcx, [nm1]
    mov rdx, rbx
    mov r8, r12
    mov r9d, 4
    call par_for
    ; the threads' sums, in thread order
    lea rdi, [nm_tot]
    mov ecx, NMS
    xor eax, eax
    rep stosq
    xor edx, edx
.t:
    cmp rdx, [nthreads]
    jae .tot
    imul r8, rdx, NMS * 8
    add r8, [nm_sum]
    xor ecx, ecx
.tb:
    lea rax, [nm_tot]
    movsd xmm0, [rax+rcx*8]
    addsd xmm0, [r8+rcx*8]
    movsd [rax+rcx*8], xmm0
    inc ecx
    cmp ecx, 161
    jb .tb
    inc rdx
    jmp .t
.tot:
    lea r8, [nm_tot]
    movsd xmm4, [r8+160*8]      ; frames
    xor ecx, ecx
.m:
    movsd xmm0, [r8+rcx*8]
    divsd xmm0, xmm4            ; mean
    movsd xmm1, [r8+rcx*8+MEL_N*8]
    divsd xmm1, xmm4
    movapd xmm2, xmm0
    mulsd xmm2, xmm0
    subsd xmm1, xmm2            ; variance
    addsd xmm1, [c_epsv]
    sqrtsd xmm1, xmm1
    movsd xmm2, [c_one]
    divsd xmm2, xmm1
    cvtsd2ss xmm0, xmm0
    movss [rbx+LD_MEAN+rcx*4], xmm0
    cvtsd2ss xmm2, xmm2
    movss [rbx+LD_ISTD+rcx*4], xmm2
    inc ecx
    cmp ecx, MEL_N
    jb .m
    mov rcx, r13
    call file_create
    mov rsi, rax
    mov rcx, rax
    lea rdx, [rbx+LD_MEAN]
    mov r8d, MEL_N * 8
    call file_write
    mov rcx, rsi
    call file_close
    mov rcx, [nm_sum]
    call mem_free
    mov rcx, [nm_ids]
    call mem_free
    mov eax, 1
.out:
    add rsp, 48
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .bss
nm_tot  resq NMS
section .text

; ---- ld_plan

; eax = samples -> eax = its padded mel frames (flags in [pl_flags])
padded:
    add eax, MEL_HOP - 1
    xor edx, edx
    mov ecx, MEL_HOP
    div ecx
    test dword [pl_flags], PL_FAST
    jz .eight
    add eax, 127
    and eax, -128
    ret
.eight:
    add eax, 7
    and eax, -8
    ret

; rcx = order entries, edx = how many, r8 = LD's table. sorts them by samples
; (shell sort, Ciura's gaps)
sortn:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    mov r9, rcx
    mov r10d, edx
    lea r11, [gaps]
.gap:
    mov r12d, [r11]
    test r12d, r12d
    jz .done
    add r11, 4
    mov esi, r12d               ; i
.i:
    cmp esi, r10d
    jae .gap
    mov r13d, [r9+rsi*4]        ; tmp
    imul rax, r13, UR_SIZE
    mov r14d, [r8+rax+UR_N]     ; its key
    mov edi, esi                ; j
.j:
    cmp edi, r12d
    jb .put
    mov ebx, edi
    sub ebx, r12d
    mov eax, [r9+rbx*4]
    imul rcx, rax, UR_SIZE
    cmp [r8+rcx+UR_N], r14d
    jbe .put
    mov [r9+rdi*4], eax
    mov edi, ebx
    jmp .j
.put:
    mov [r9+rdi*4], r13d
    inc esi
    jmp .i
.done:
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .rdata
align 4
gaps    dd 1750, 701, 301, 132, 57, 23, 10, 4, 1, 0
section .text

; rax = bound. rax = a random number below it
below:
    push rbx
    sub rsp, 32
    mov rbx, rax
    lea rcx, [rng]
    call rng_next
    xor edx, edx
    div rbx
    mov rax, rdx
    add rsp, 32
    pop rbx
    ret

; rcx = LD (indexed), rdx = mel frames a micro-batch may hold, r8 = most utterances
; in one, r9 = seed, [rsp+40] = flags (PL_*). an epoch's plan: the utterances
; shuffled, then sorted by length a pool at a time and cut into micro-batches as
; full as the budget allows, the micro-batches shuffled. PL_FAST: T a multiple of
; 128, B of 8 (a pool's last few that can't make 8 sit this epoch out).
; rax = micro-batches
global ld_plan
ld_plan:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov rbx, rcx
    mov [pl_budget], rdx
    mov [pl_maxb], r8
    mov rax, [rsp+144]
    mov [pl_flags], rax
    lea rcx, [rng]
    mov rdx, r9
    call rng_seed
    cmp qword [rbx+LD_ORDER], 0
    jne .have
    mov rcx, [rbx+LD_NU]
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov [rbx+LD_ORDER], rax
    mov rcx, [rbx+LD_NU]
    imul rcx, MP_SIZE
    add rcx, 64
    call mem_alloc
    mov [rbx+LD_B], rax
.have:
    mov rdi, [rbx+LD_ORDER]
    xor eax, eax
.id:
    cmp rax, [rbx+LD_NU]
    jae .shuffle
    mov [rdi+rax*4], eax
    inc rax
    jmp .id
.shuffle:
    mov rsi, [rbx+LD_NU]
.fy:
    cmp rsi, 1
    jbe .pools
    mov rax, rsi
    call below
    dec rsi
    mov ecx, [rdi+rsi*4]
    mov r8d, [rdi+rax*4]
    mov [rdi+rsi*4], r8d
    mov [rdi+rax*4], ecx
    jmp .fy
.pools:
    xor r12d, r12d              ; pool start
    xor r15d, r15d              ; micro-batches
.pool:
    cmp r12, [rbx+LD_NU]
    jae .cut
    mov r13, r12
    add r13, POOL
    cmp r13, [rbx+LD_NU]
    jbe .pe
    mov r13, [rbx+LD_NU]
.pe:
    lea rcx, [rdi+r12*4]
    mov edx, r13d
    sub edx, r12d
    mov r8, [rbx+LD_U]
    call sortn
    mov rsi, r12                ; i
.mb:
    cmp rsi, r13
    jae .pn
    ; as many as fit: sorted, so the last one sets T
    xor r14d, r14d              ; c
.grow:
    lea rax, [rsi+r14]
    cmp rax, r13
    jae .grown
    mov eax, [rdi+rax*4]
    imul rax, UR_SIZE
    add rax, [rbx+LD_U]
    mov eax, [rax+UR_N]
    call padded
    lea rcx, [r14+1]
    imul rax, rcx
    cmp rax, [pl_budget]
    ja .grown
    cmp rcx, [pl_maxb]
    ja .grown
    inc r14
    jmp .grow
.grown:
    test dword [pl_flags], PL_FAST
    jz .any
    cmp r14, 8
    jae .eights
    ; under 8: the pool's tail (sits out), or too long to take 8 (left out)
    lea rax, [rsi+r14]
    cmp rax, r13
    jae .pn
    inc rsi
    jmp .mb
.eights:
    and r14, -8
    jmp .emit
.any:
    test r14, r14
    jnz .emit
    inc r14                     ; longer than the budget alone: it goes by itself
.emit:
    imul r8, r15, MP_SIZE
    add r8, [rbx+LD_B]
    mov [r8+MP_START], esi
    mov [r8+MP_B], r14d
    lea rax, [rsi+r14-1]
    mov eax, [rdi+rax*4]
    imul rax, UR_SIZE
    add rax, [rbx+LD_U]
    mov eax, [rax+UR_N]
    call padded
    mov [r8+MP_T], eax
    ; the most tokens
    xor eax, eax
    xor ecx, ecx
.mu:
    cmp rcx, r14
    jae .mud
    lea rdx, [rsi+rcx]
    mov edx, [rdi+rdx*4]
    imul rdx, UR_SIZE
    add rdx, [rbx+LD_U]
    movzx edx, word [rdx+UR_NT]
    cmp edx, eax
    cmova eax, edx
    inc rcx
    jmp .mu
.mud:
    mov [r8+MP_U], eax
    inc r15
    add rsi, r14
    jmp .mb
.pn:
    mov r12, r13
    jmp .pool
.cut:
    ; shuffle the micro-batches
    mov rsi, r15
    mov rdi, [rbx+LD_B]
.fyb:
    cmp rsi, 1
    jbe .done
    mov rax, rsi
    call below
    dec rsi
    imul rcx, rsi, MP_SIZE
    shl rax, 4
    mov r8, [rdi+rcx]
    mov r9, [rdi+rcx+8]
    mov r10, [rdi+rax]
    mov r11, [rdi+rax+8]
    mov [rdi+rcx], r10
    mov [rdi+rcx+8], r11
    mov [rdi+rax], r8
    mov [rdi+rax+8], r9
    jmp .fyb
.done:
    mov [rbx+LD_NB], r15
    mov rax, r15
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

section .bss
pl_budget resq 1
pl_maxb   resq 1
pl_flags  resq 1
section .text

; ---- ld_fill

; rcx = ctx, rdx = start, r8 = end, r9 = thread: utterances [start, end) of the
; micro-batch in fl_mp into the batch at fl_hb
fill1:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov rsi, rdx
    mov [rsp+40], r8            ; end
    mov r12, r9
.u:
    cmp rsi, [rsp+40]
    jae .done
    mov r14, [fl_ld]
    mov r15, [fl_hb]
    mov rax, [fl_mp]
    mov eax, [rax+MP_START]
    add rax, rsi
    mov rcx, [r14+LD_ORDER]
    mov eax, [rcx+rax*4]
    imul r13, rax, UR_SIZE
    add r13, [r14+LD_U]         ; the utterance
    mov rcx, r14
    mov rdx, r13
    mov r8, r12
    call decode
    mov rdi, rax                ; samples
    mov eax, [r15+HB_T]
    imul rax, rsi
    imul rbx, rax, MEL_N * 4
    add rbx, [r15+HB_MEL]       ; its rows
    mov rcx, r12
    call pcmbuf
    mov rdx, rax
    lea rcx, [mel]
    mov r8, rdi
    mov r9, rbx
    call mel_run
    ; encoder frames, then the rows normalized and zeros to T
    mov rcx, [r15+HB_TL]
    lea edx, [rax+7]
    shr edx, 3
    mov [rcx+rsi*4], edx
    mov r9, rax                 ; frames
    mov r8, rbx
    mov r10, rax
.f:
    test r9, r9
    jz .pad
    xor ecx, ecx
.b:
    movss xmm0, [r8+rcx*4]
    subss xmm0, [r14+LD_MEAN+rcx*4]
    mulss xmm0, [r14+LD_ISTD+rcx*4]
    movss [r8+rcx*4], xmm0
    inc ecx
    cmp ecx, MEL_N
    jb .b
    add r8, MEL_N * 4
    dec r9
    jmp .f
.pad:
    mov ecx, [r15+HB_T]
    sub rcx, r10
    imul rcx, MEL_N
    mov rdi, r8
    xor eax, eax
    rep stosd
    ; labels: the tokens, then zeros to U
    mov eax, [r15+HB_U]
    imul rax, rsi
    shl rax, 2
    add rax, [r15+HB_LAB]
    movzx ecx, word [r13+UR_NT]
    mov rdx, [r15+HB_UL]
    mov [rdx+rsi*4], ecx
    mov edx, [r13+UR_TOK]
    mov r8, [r14+LD_TOK]
    lea r8, [r8+rdx*2]
    xor edx, edx
.l:
    cmp edx, [r15+HB_U]
    jae .next
    xor r9d, r9d
    cmp edx, ecx
    jae .z
    movzx r9d, word [r8+rdx*2]
.z:
    mov [rax+rdx*4], r9d
    inc edx
    jmp .l
.next:
    inc rsi
    jmp .u
.done:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = LD (planned), edx = which micro-batch, r8 = HB (its buffers big enough).
; decodes, computes the log-mel and fills the batch, every thread on it
global ld_fill
ld_fill:
    sub rsp, 40
    mov [fl_ld], rcx
    mov [fl_hb], r8
    mov eax, edx
    imul rax, MP_SIZE
    add rax, [rcx+LD_B]
    mov [fl_mp], rax
    mov ecx, [rax+MP_B]
    mov [r8+HB_B], ecx
    mov ecx, [rax+MP_T]
    mov [r8+HB_T], ecx
    mov ecx, [rax+MP_U]
    mov [r8+HB_U], ecx
    lea rcx, [fill1]
    xor edx, edx
    mov r8d, [rax+MP_B]
    mov r9d, 1
    call par_for
    mov r8, [fl_hb]
    mov rcx, [r8+HB_UL]
    xor eax, eax
    xor edx, edx
.n:
    cmp edx, [r8+HB_B]
    jae .nd
    add eax, [rcx+rdx*4]
    inc edx
    jmp .n
.nd:
    mov [r8+HB_NLAB], eax
    add rsp, 40
    ret
