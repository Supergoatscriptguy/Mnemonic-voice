; shards DIR [check=1] [show=5] [play=0] [seed=N]
; the dataset inspector. checks every utterance in DIR\*.shard (each flac decoded,
; md5 and crcs checked, length against the index) on every thread, prints the totals,
; then shows a few at random with their transcripts. play=N plays N random ones
; uses: data\shard audio\flac audio\dev audio\resample audio\wav
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess, Sleep

MAXSH   equ 4096

section .rdata
k_check db "check", 0
k_show  db "show", 0
k_play  db "play", 0
k_seed  db "seed", 0
s_pat   db "\*.shard", 0
usage   db "usage: shards DIR [check=1] [show=5] [play=0] [seed=N]", 13, 10, 0
e_none  db "no shards there", 0
align 8
c_rate  dq 16000.0
c_hour  dq 3600.0

section .bss
alignb 8
pattern resb 1024
found   resb 1 << 20
paths   resq 1
nsh     resq 1
srs     resb SR_SIZE * MAXSH
total   resq 1                  ; utterances
samples resq 1
bytes   resq 1
bad     resq 1
longest resq 1
ra      resb RNG_SIZE
fd      resb FD_SIZE
clip    resb CL_SIZE
conv    resb CL_SIZE
dev     resb DV_SIZE

section .text

; par_for callback: shards rdx..r8, decodes every utterance
check:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40 + FD_SIZE + 8
    mov rsi, rdx
    mov rdi, r8
.shard:
    cmp rsi, rdi
    jae .out
    imul rbx, rsi, SR_SIZE
    lea rax, [srs]
    add rbx, rax
    xor r12d, r12d
.utt:
    cmp r12, [rbx+SR_COUNT]
    jae .nextshard
    imul r13, r12, UT_SIZE
    add r13, [rbx+SR_INDEX]
    mov rcx, [r13+UT_FLAC]
    add rcx, [rbx+SR_BASE]
    mov edx, [r13+UT_FLEN]
    lea r8, [rsp+40]
    call flac_decode
    mov r14d, 1                 ; bad until it checks out
    cmp eax, 1
    jne .verdict
    cmp dword [rsp+40+FD_MD5], 1
    jne .free
    mov eax, [r13+UT_N]
    cmp rax, [rsp+40+FD_N]
    jne .free
    xor r14d, r14d
.free:
    mov rcx, [rsp+40+FD_PCM]
    call mem_free
.verdict:
    test r14d, r14d
    jz .good
    lock inc qword [bad]
.good:
    mov eax, [r13+UT_N]
    lock add [samples], rax
    mov eax, [r13+UT_FLEN]
    lock add [bytes], rax
    lock inc qword [total]
    mov edx, [r13+UT_N]
.longer:
    mov rax, [longest]
    cmp rdx, rax
    jbe .notlonger
    lock cmpxchg [longest], rdx
    jnz .longer                 ; someone else got there first, look again
.notlonger:
    inc r12
    jmp .utt
.nextshard:
    inc rsi
    jmp .shard
.out:
    add rsp, 40 + FD_SIZE + 8
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = text, edx = length. prints it, or (none)
ptext:
    test edx, edx
    jnz print
    sub rsp, 40
    say "(none)"
    add rsp, 40
    ret

; picks a random utterance. rbx = its SR, rsi = its UT, rdi = the shard's path
pick:
    sub rsp, 40
.again:
    lea rcx, [ra]
    call rng_next
    xor edx, edx
    div qword [nsh]
    imul rbx, rdx, SR_SIZE
    lea rax, [srs]
    add rbx, rax
    mov rax, [paths]
    mov rdi, [rax+rdx*8]
    cmp qword [rbx+SR_COUNT], 0
    je .again
    lea rcx, [ra]
    call rng_next
    xor edx, edx
    div qword [rbx+SR_COUNT]
    imul rsi, rdx, UT_SIZE
    add rsi, [rbx+SR_INDEX]
    add rsp, 40
    ret

; shows the utterance pick chose
show:
    sub rsp, 40
    say "  "
    mov rcx, rdi
    call print_z
    say "  "
    mov rcx, [rsi+UT_ID]
    add rcx, [rbx+SR_BASE]
    mov edx, [rsi+UT_ILEN]
    call ptext
    say ", speaker "
    mov ecx, [rsi+UT_SPK]
    call print_dec
    say ", "
    mov eax, [rsi+UT_N]
    cvtsi2sd xmm0, rax
    divsd xmm0, [c_rate]
    mov edx, 2
    call print_fixed
    say " s", 13, 10, "    "
    mov rcx, [rsi+UT_TEXT]
    add rcx, [rbx+SR_BASE]
    mov edx, [rsi+UT_TLEN]
    call ptext
    say 13, 10, "    "
    mov rcx, [rsi+UT_NORM]
    add rcx, [rbx+SR_BASE]
    mov edx, [rsi+UT_NLEN]
    call ptext
    say 13, 10
    add rsp, 40
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    call lib_init
    cmp qword [argc], 2
    jae .go
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
.go:
    lea rax, [argv]
    mov rdx, [rax+8]
    lea rcx, [pattern]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_pat]
    call fmt_str
    mov byte [rax], 0
    call cfg_args
    lea rcx, [pattern]
    lea rdx, [found]
    mov r8d, 1 << 20
    call file_find
    test rax, rax
    jnz .some
    lea rcx, [e_none]
    call fatal
.some:
    cmp rax, MAXSH
    jbe .fits
    mov eax, MAXSH
.fits:
    mov [nsh], rax
    mov [paths], rdx
    xor r12d, r12d
.open:
    cmp r12, [nsh]
    jae .opened
    mov rax, [paths]
    mov rcx, [rax+r12*8]
    imul rdx, r12, SR_SIZE
    lea rax, [srs]
    add rdx, rax
    call shard_open
    test eax, eax
    jnz .ok
    say "  not a shard: "
    mov rax, [paths]
    mov rcx, [rax+r12*8]
    call print_z
    say 13, 10
    imul rdx, r12, SR_SIZE
    lea rax, [srs]
    mov qword [rax+rdx+SR_COUNT], 0
.ok:
    inc r12
    jmp .open
.opened:
    lea rcx, [k_check]
    mov edx, 1
    call cfg_int
    test rax, rax
    jz .shows
    xor ecx, ecx
    call pool_init
    call time_now
    mov [rsp+32], rax
    lea rcx, [check]
    xor edx, edx
    mov r8, [nsh]
    mov r9d, 1
    call par_for
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+32], xmm0
    mov rcx, [nsh]
    call print_dec
    say " shards, "
    mov rcx, [total]
    call print_dec
    say " utterances, "
    cvtsi2sd xmm0, qword [samples]
    divsd xmm0, [c_rate]
    divsd xmm0, [c_hour]
    mov edx, 2
    call print_fixed
    say " h, "
    mov rcx, [bytes]
    call print_dec
    say " bytes of flac, longest "
    cvtsi2sd xmm0, qword [longest]
    divsd xmm0, [c_rate]
    mov edx, 1
    call print_fixed
    say " s", 13, 10, "checked in "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " s: "
    mov rcx, [bad]
    call print_dec
    say " bad", 13, 10

.shows:
    call time_now
    mov rdx, rax
    lea rcx, [k_seed]
    call cfg_int
    mov rdx, rax
    lea rcx, [ra]
    call rng_seed
    lea rcx, [k_show]
    mov edx, 5
    call cfg_int
    mov r12, rax
.show:
    test r12, r12
    jz .plays
    call pick
    call show
    dec r12
    jmp .show

.plays:
    lea rcx, [k_play]
    xor edx, edx
    call cfg_int
    mov r12, rax
    test r12, r12
    jz .bye
    xor ecx, ecx
    lea rdx, [dev]
    mov r8d, 100
    call dev_open
    test eax, eax
    jz .bye
.play:
    test r12, r12
    jz .closedev
    call pick
    call show
    mov rcx, [rsi+UT_FLAC]
    add rcx, [rbx+SR_BASE]
    mov edx, [rsi+UT_FLEN]
    lea r8, [fd]
    call flac_decode
    test eax, eax
    jz .skip
    ; int16 -> a float clip, then the device's rate
    mov rcx, [fd+FD_N]
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov [clip+CL_DATA], rax
    mov rcx, [fd+FD_N]
    mov [clip+CL_FRAMES], rcx
    mov dword [clip+CL_RATE], 16000
    mov dword [clip+CL_CH], 1
    mov rdx, [fd+FD_PCM]
    mov r9, [clip+CL_DATA]
    mov r8d, 0x38000000         ; 2^-15
    movd xmm1, r8d
    test rcx, rcx
    jz .floated
.f:
    movsx eax, word [rdx+rcx*2-2]
    cvtsi2ss xmm0, eax
    mulss xmm0, xmm1
    movss [r9+rcx*4-4], xmm0
    dec rcx
    jnz .f
.floated:
    lea rcx, [clip]
    mov edx, [dev+DV_RATE]
    lea r8, [conv]
    call clip_resample
    lea rcx, [dev]
    lea rdx, [conv]
    call dev_play
    mov rcx, [conv+CL_DATA]
    call mem_free
    mov rcx, [clip+CL_DATA]
    call mem_free
    mov rcx, [fd+FD_PCM]
    call mem_free
    ; a fresh stream for the next one
    lea rcx, [dev]
    call dev_close
    lea rdi, [dev]
    xor eax, eax
    mov ecx, DV_SIZE
    rep stosb
    xor ecx, ecx
    lea rdx, [dev]
    mov r8d, 100
    call dev_open
    mov ecx, 400
    call Sleep
.skip:
    dec r12
    jmp .play
.closedev:
    lea rcx, [dev]
    call dev_close
.bye:
    call con_restore
    mov rcx, [bad]
    call ExitProcess
