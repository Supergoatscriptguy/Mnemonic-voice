; stage 3 test: md5 against rfc 1321, then every flac in the librispeech test sets and
; the libri-light sample decoded and checked against the md5 its encoder stored
; uses: audio\flac data\parquet data\snappy data\zstd data\tar
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "data/parquet.inc"
%include "data/tar.inc"
%include "audio/audio.inc"

section .rdata
m0      db ""
m1      db "a"
m2      db "abc"
m3      db "message digest"
m4      db "abcdefghijklmnopqrstuvwxyz"
m5      db "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
m6      db "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
x0      db 0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e
x1      db 0x0c, 0xc1, 0x75, 0xb9, 0xc0, 0xf1, 0xb6, 0xa8, 0x31, 0xc3, 0x99, 0xe2, 0x69, 0x77, 0x26, 0x61
x2      db 0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1, 0x7f, 0x72
x3      db 0xf9, 0x6b, 0x69, 0x7d, 0x7c, 0xb7, 0x93, 0x8d, 0x52, 0x5a, 0x2f, 0x31, 0xaa, 0xf1, 0x61, 0xd0
x4      db 0xc3, 0xfc, 0xd3, 0xd7, 0x61, 0x92, 0xe4, 0x00, 0x7d, 0xfb, 0x49, 0x6c, 0xca, 0x67, 0xe1, 0x3b
x5      db 0xd1, 0x74, 0xab, 0x98, 0xd2, 0x77, 0xd9, 0xf5, 0xa5, 0x61, 0x1c, 0x2c, 0x9f, 0x41, 0x9d, 0x9f
x6      db 0x57, 0xed, 0xf4, 0xa2, 0x2b, 0xe3, 0xc9, 0x55, 0xac, 0x49, 0xda, 0x2e, 0x21, 0x07, 0xb6, 0x7a
f_tc    db "datasets\librispeech\test.clean.parquet", 0
f_to    db "datasets\librispeech\test.other.parquet", 0
f_ll    db "datasets\librilight\small.head", 0
p_vec   db "test\flac\*.flac", 0
l_vec   db "test\flac (made by ffmpeg)", 0
c_audio db "audio.bytes", 0
s_flac  db ".flac", 0
align 8
c_hour  dq 3600.0
c_rate  dq 16000.0
c_m     dq 1e-6

section .bss
alignb 8
fd      resb FD_SIZE
pq      resb PQ_SIZEOF
ar      resb ARENA_SIZE
col     resb COL_SIZE
ta      resb TA_SIZEOF
digest  resb 16
stats   resq 16
nok     resq 1
nbad    resq 1
nsamp   resq 1
ticks   resq 1
found   resb 8192

section .text

; rcx = text, edx = length, r8 = expected digest. eax = 1 if md5 gives it
md5is:
    push rbx
    sub rsp, 32
    mov rbx, r8
    lea r8, [digest]
    call md5
    mov rax, [digest]
    cmp rax, [rbx]
    jne .no
    mov rax, [digest+8]
    cmp rax, [rbx+8]
    jne .no
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop rbx
    ret

%macro md5check 3
    lea rcx, [%1]
    mov edx, %2
    lea r8, [%3]
    call md5is
    cmp eax, 1
%endmacro

; rcx = flac bytes, rdx = size. decodes it, checks md5 and length, adds up the stats
one:
    push rbx
    sub rsp, 48
    mov [rsp+32], rcx
    mov [rsp+40], rdx
    call time_now
    mov rbx, rax
    mov rcx, [rsp+32]
    mov rdx, [rsp+40]
    lea r8, [fd]
    call flac_decode
    mov [rsp+32], eax
    call time_now
    sub rax, rbx
    add [ticks], rax
    cmp dword [rsp+32], 1
    jne .bad
    cmp dword [fd+FD_MD5], 1
    jne .bad
    mov rax, [fd+FD_N]
    cmp rax, [fd+FI_SAMPLES]
    jne .bad
    add [nsamp], rax
    inc qword [nok]
    xor ecx, ecx
    lea r8, [fd]
.st:
    mov rax, [r8+FD_STATS+rcx*8]
    lea rdx, [stats]
    cmp ecx, 12
    je .max
    add [rdx+rcx*8], rax
    jmp .nx
.max:
    cmp rax, [rdx+rcx*8]
    jbe .nx
    mov [rdx+rcx*8], rax
.nx:
    inc ecx
    cmp ecx, 13
    jb .st
    jmp .free
.bad:
    inc qword [nbad]
    say "    failed: decode "
    mov ecx, [rsp+32]
    call print_dec
    say ", md5 "
    mov ecx, [fd+FD_MD5]
    call print_dec
    say ", got "
    mov rcx, [fd+FD_N]
    call print_dec
    say " of "
    mov rcx, [fd+FI_SAMPLES]
    call print_dec
    say " samples, "
    mov rcx, [fd+FD_STATS+11*8]
    call print_dec
    say " frames, max block "
    mov ecx, [fd+FI_MAXBS]
    call print_dec
    say ", stats"
    xor ebx, ebx
.ps:
    say " "
    lea rax, [fd]
    mov rcx, [rax+FD_STATS+rbx*8]
    call print_dec
    inc ebx
    cmp ebx, 14
    jb .ps
    say 13, 10
.free:
    mov rcx, [fd+FD_PCM]
    test rcx, rcx
    jz .out
    call mem_free
.out:
    add rsp, 48
    pop rbx
    ret

; rcx = parquet path. every audio.bytes value through one
parquet:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rdx, rcx
    lea rcx, [pq]
    call pq_open
    test eax, eax
    jz .out
    lea rcx, [pq]
    lea rdx, [c_audio]
    call pq_leaf
    mov r12, rax
    xor ebx, ebx
.rg:
    cmp rbx, [pq+PQ_NRG]
    jae .closed
    lea rcx, [pq]
    lea rdx, [col]
    mov r8, rbx
    mov r9, r12
    call pq_chunk
    xor edi, edi
.v:
    cmp rdi, [col+COL_NV]
    jae .rgdone
    mov rax, [col+COL_VALS]
    mov rdx, rdi
    shl rdx, 4
    mov rcx, [rax+rdx]
    mov rdx, [rax+rdx+8]
    call one
    inc rdi
    jmp .v
.rgdone:
    lea rcx, [ar]
    call arena_reset
    inc rbx
    jmp .rg
.closed:
    lea rcx, [pq]
    call pq_close
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; prints the totals so far and checks there were no failures. rcx = label
report:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    say "  "
    mov rcx, rbx
    call print_z
    say ": "
    mov rcx, [nok]
    call print_dec
    say " streams, "
    cvtsi2sd xmm0, qword [nsamp]
    divsd xmm0, [c_rate]
    divsd xmm0, [c_hour]
    mov edx, 2
    call print_fixed
    say " h, decoded at "
    mov rcx, [ticks]
    cvtsi2sd xmm1, rcx
    cvtsi2sd xmm0, qword [qpc_freq]
    divsd xmm1, xmm0            ; seconds
    cvtsi2sd xmm0, qword [nsamp]
    divsd xmm0, xmm1
    mulsd xmm0, [c_m]
    mov edx, 0
    call print_fixed
    say "M samples/s on one thread", 13, 10
    cmp qword [nbad], 0
    check e, "    every one decodes, matches its md5 and its length"
    add rsp, 32
    pop rbx
    ret

%macro statline 2
    say %2
    mov rcx, [stats+%1*8]
    call print_dec
%endmacro

; what subframes and partitions turned up so far, then clears the counts
seen:
    sub rsp, 40
    say "  subframes seen: constant "
    statline 0, ""
    statline 1, ", verbatim "
    statline 2, ", fixed order 0 "
    statline 3, " / 1 "
    statline 4, " / 2 "
    statline 5, " / 3 "
    statline 6, " / 4 "
    statline 7, ", lpc "
    statline 12, " (orders up to "
    say ")", 13, 10, "  partitions: rice2 "
    statline 8, ""
    statline 9, ", escaped "
    statline 10, ", wasted bits subframes "
    statline 13, ", long unary codes "
    statline 11, ", frames "
    say 13, 10
    lea rdi, [stats]
    xor eax, eax
    mov ecx, 16
    rep stosq
    add rsp, 40
    ret


global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init

    say "md5 (rfc 1321)", 13, 10
    md5check m0, 0, x0
    check e, '  ""'
    md5check m1, 1, x1
    check e, '  "a"'
    md5check m2, 3, x2
    check e, '  "abc"'
    md5check m3, 14, x3
    check e, '  "message digest"'
    md5check m4, 26, x4
    check e, "  a-z"
    md5check m5, 62, x5
    check e, "  A-Z a-z 0-9 (62 bytes, the padding needs a second block)"
    md5check m6, 80, x6
    check e, "  1234567890 x 8 (80 bytes)"

    lea rcx, [ar]
    mov rdx, 1 << 34
    call arena_init
    lea rax, [ar]
    mov [col+COL_ARENA], rax
    mov ecx, ZSTD_CTX
    call mem_alloc
    mov [col+COL_ZSTD], rax
    mov rcx, rax
    call zstd_init

    say "flac, every stream checked against its own md5", 13, 10
    lea rcx, [f_tc]
    call file_exists
    test eax, eax
    jz .nols
    lea rcx, [f_tc]
    call parquet
    lea rcx, [f_to]
    call parquet
    lea rcx, [f_tc+9]
    call report
    jmp .ll
.nols:
    say "  (no librispeech test sets in datasets\, skipped)", 13, 10

.ll:
    lea rcx, [f_ll]
    call file_map
    test rax, rax
    jz .noll
    mov qword [nok], 0
    mov qword [nsamp], 0
    mov qword [ticks], 0
    lea rcx, [ta]
    mov r8, rdx
    mov rdx, rax
    call tar_open
.member:
    lea rcx, [ta]
    call tar_next
    cmp eax, 1
    jne .lldone                 ; the end, or the one the head cuts off
    ; .flac files only
    mov rcx, [ta+TA_NAME]
.end:
    cmp byte [rcx], 0
    je .ext
    inc rcx
    jmp .end
.ext:
    cmp dword [rcx-4], 'flac'
    jne .member
    mov rcx, [ta+TA_DATA]
    mov rdx, [ta+TA_SIZE]
    call one
    jmp .member
.lldone:
    lea rcx, [f_ll+9]
    call report
    jmp .stats
.noll:
    say "  (no libri-light sample, skipped)", 13, 10

.stats:
    call seen
    ; vectors from another encoder, to reach what the datasets never use
    mov qword [nok], 0
    mov qword [nsamp], 0
    mov qword [ticks], 0
    lea rcx, [p_vec]
    lea rdx, [found]
    mov r8d, 8192
    call file_find
    mov rbx, rax
    mov rsi, rdx
    cmp rbx, 5
    check ae, "  the reference vectors are there"
.vec:
    test rbx, rbx
    jz .vecdone
    mov rcx, [rsi]
    call file_read_all
    test rax, rax
    jz .vnext
    mov rdi, rax
    mov rcx, rax
    call one
    mov rcx, rdi
    call mem_free
.vnext:
    add rsi, 8
    dec rbx
    jmp .vec
.vecdone:
    lea rcx, [l_vec]
    call report
    call seen
    jmp t_done
