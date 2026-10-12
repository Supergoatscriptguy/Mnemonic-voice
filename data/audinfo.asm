; audinfo file.parquet [column]   what the audio blobs in a column look like (default audio.bytes)
; audinfo file.tar                 or the files in a tar (a cut off one is fine)
; counts every distinct format: codec, rate, channels, bits, and for flac the block
; sizes, metadata blocks and whether the md5 is set. with total hours
; uses: data\parquet data\snappy data\zstd data\tar audio\flac audio\wav
default rel
bits 64
%include "lib.inc"
%include "data/parquet.inc"
%include "data/tar.inc"
%include "audio/audio.inc"

extern ExitProcess

MAXK    equ 64
K_TEXT  equ 0                   ; the format, as text
K_N     equ 112
K_SECS  equ 120
K_SIZE  equ 128

section .rdata
align 8
c_hour  dq 3600.0
d_col   db "audio.bytes", 0
usage   db "usage: audinfo file.parquet [column] | file.tar", 13, 10, 0
e_open  db "can't open that file", 0
e_col   db "no such column", 0

section .bss
alignb 8
pq      resb PQ_SIZEOF
ar      resb ARENA_SIZE
col     resb COL_SIZE
ta      resb TA_SIZEOF
fi      resb FI_SIZE
clip    resb CL_SIZE
keys    resb K_SIZE * MAXK
nkeys   resq 1
key     resb 128
blobs   resq 1
named   resq 1

section .text

; rcx = text. finds or adds it, counts one more, adds xmm0 seconds
tally:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    lea rbx, [keys]
    xor edi, edi
.look:
    cmp rdi, [nkeys]
    jae .add
    mov rcx, rsi
    mov rdx, rbx
.cmp:
    mov al, [rcx]
    cmp al, [rdx]
    jne .other
    inc rcx
    inc rdx
    test al, al
    jnz .cmp
    jmp .hit
.other:
    add rbx, K_SIZE
    inc rdi
    jmp .look
.add:
    cmp rdi, MAXK
    jae .out
    inc qword [nkeys]
    mov rcx, rbx
    mov rdx, rsi
    call fmt_str
    mov byte [rax], 0
.hit:
    inc qword [rbx+K_N]
    addsd xmm0, [rbx+K_SECS]
    movsd [rbx+K_SECS], xmm0
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = bytes, rdx = size, r8 = file name or 0. works out the format into key,
; xmm0 = seconds of audio (0 if it isn't audio)
classify:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rcx
    mov rbx, rdx
    mov r12, r8
    lea rdi, [key]
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [fi]
    call flac_info
    test eax, eax
    jz .notflac
    emit "flac "
    mov rcx, rdi
    mov edx, [fi+FI_RATE]
    call fmt_dec
    mov rdi, rax
    emit " Hz "
    mov rcx, rdi
    mov edx, [fi+FI_CH]
    call fmt_dec
    mov rdi, rax
    emit " ch "
    mov rcx, rdi
    mov edx, [fi+FI_BITS]
    call fmt_dec
    mov rdi, rax
    emit " bit, blocks "
    mov rcx, rdi
    mov edx, [fi+FI_MINBS]
    call fmt_dec
    mov rdi, rax
    mov edx, [fi+FI_MAXBS]
    cmp edx, [fi+FI_MINBS]
    je .meta
    mov byte [rdi], '-'
    lea rcx, [rdi+1]
    call fmt_dec
    mov rdi, rax
.meta:
    emit ", meta"
    mov ebx, [fi+FI_BLOCKS]
    btr ebx, 0                  ; streaminfo is always there
.mb:
    test ebx, ebx
    jz .md5
    bsf edx, ebx
    btr ebx, edx
    mov byte [rdi], ' '
    lea rcx, [rdi+1]
    call fmt_dec
    mov rdi, rax
    jmp .mb
.md5:
    mov rax, [fi+FI_MD5]
    or rax, [fi+FI_MD5+8]
    jnz .md5set
    emit ", no md5"
    jmp .flacsecs
.md5set:
    emit ", md5"
.flacsecs:
    cvtsi2sd xmm0, qword [fi+FI_SAMPLES]
    cvtsi2sd xmm1, dword [fi+FI_RATE]
    divsd xmm0, xmm1
    jmp .done

.notflac:
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [clip]
    call wav_head
    test eax, eax
    jz .notwav
    emit "wav "
    cmp dword [clip+CL_FLOAT], 0
    je .pcm
    emit "float"
    jmp .wbits
.pcm:
    emit "pcm"
.wbits:
    mov rcx, rdi
    mov edx, [clip+CL_BITS]
    call fmt_dec
    mov rdi, rax
    mov byte [rdi], ' '
    lea rcx, [rdi+1]
    mov edx, [clip+CL_RATE]
    call fmt_dec
    mov rdi, rax
    emit " Hz "
    mov rcx, rdi
    mov edx, [clip+CL_CH]
    call fmt_dec
    mov rdi, rax
    emit " ch"
    cvtsi2sd xmm0, qword [clip+CL_FRAMES]
    cvtsi2sd xmm1, dword [clip+CL_RATE]
    divsd xmm0, xmm1
    jmp .done

.notwav:
    xorpd xmm0, xmm0
    cmp rbx, 4
    jb .ext
    mov eax, [rsi]
    cmp eax, 'OggS'
    jne .id3
    emit "ogg"
    jmp .done
.id3:
    and eax, 0xffffff
    cmp eax, 'ID3'
    je .mp3
    movzx eax, word [rsi]
    and eax, 0xe0ff             ; mpeg frame sync, 11 bits
    cmp eax, 0xe0ff
    jne .ext
.mp3:
    emit "mp3"
    jmp .done
.ext:
    ; not audio: count it by the name's extension, if there's a name
    test r12, r12
    jz .unknown
    mov rdx, r12
    xor ebx, ebx                ; the last dot (emit trashes rcx)
.dot:
    mov al, [rdx]
    test al, al
    jz .gotdot
    cmp al, '.'
    jne .nd
    mov rbx, rdx
.nd:
    inc rdx
    jmp .dot
.gotdot:
    test rbx, rbx
    jz .unknown
    emit "other "
    mov rdx, rbx
    mov rcx, rdi
    call fmt_str
    mov rdi, rax
    xorpd xmm0, xmm0
    jmp .done
.unknown:
    emit "unknown"
.done:
    mov byte [rdi], 0
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = bytes, rdx = size, r8 = name or 0
one:
    sub rsp, 40
    inc qword [blobs]
    call classify
    lea rcx, [key]
    call tally
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
    mov rsi, [rax+8]
    ; parquet if it says PAR1, else a tar
    mov rcx, rsi
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_open]
    call fatal
.mapped:
    mov [rsp+32], rdx
    cmp dword [rax], 'PAR1'
    je .parquet
    lea rcx, [ta]
    mov rdx, rax
    mov r8, [rsp+32]
    call .tarwalk
    jmp .report

.parquet:
    lea rcx, [pq]
    mov rdx, rsi
    call pq_open
    test eax, eax
    jnz .opened
    lea rcx, [e_open]
    call fatal
.opened:
    lea rcx, [d_col]
    cmp qword [argc], 3
    jb .colname
    lea rax, [argv]
    mov rcx, [rax+16]
.colname:
    mov rdx, rcx
    lea rcx, [pq]
    call pq_leaf
    cmp rax, -1
    jne .found
    lea rcx, [e_col]
    call fatal
.found:
    mov r12, rax
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
    xor ebx, ebx                ; row group
.rg:
    cmp rbx, [pq+PQ_NRG]
    jae .report
    lea rcx, [pq]
    lea rdx, [col]
    mov r8, rbx
    mov r9, r12
    call pq_chunk
    xor edi, edi
.val:
    cmp rdi, [col+COL_NV]
    jae .rgdone
    mov rax, [col+COL_VALS]
    mov rdx, rdi
    shl rdx, 4
    mov rcx, [rax+rdx]
    mov rdx, [rax+rdx+8]
    xor r8d, r8d
    call one
    inc rdi
    jmp .val
.rgdone:
    lea rcx, [ar]
    call arena_reset
    inc rbx
    jmp .rg

.report:
    mov rcx, rsi
    call print_z
    say ": "
    mov rcx, [blobs]
    call print_dec
    say " items", 13, 10
    lea rbx, [keys]
    xor edi, edi
.line:
    cmp rdi, [nkeys]
    jae .bye
    say "  "
    mov rcx, [rbx+K_N]
    call print_dec
    say "  "
    mov rcx, rbx
    call print_z
    xorpd xmm0, xmm0
    comisd xmm0, [rbx+K_SECS]
    je .eol
    say ", "
    movsd xmm0, [rbx+K_SECS]
    divsd xmm0, [c_hour]
    mov edx, 2
    call print_fixed
    say " h"
.eol:
    say 13, 10
    add rbx, K_SIZE
    inc rdi
    jmp .line
.bye:
    call con_restore
    xor ecx, ecx
    call ExitProcess

; rcx = TA, rdx = mapped tar, r8 = size. tallies every file, prints the first few names
.tarwalk:
    push rbx
    sub rsp, 32
    call tar_open
    xor ebx, ebx
.member:
    lea rcx, [ta]
    call tar_next
    test eax, eax
    jz .walked
    cmp ebx, 6
    jae .tally
    say "  "
    mov rcx, [ta+TA_NAME]
    call print_z
    say "  "
    mov rcx, [ta+TA_SIZE]
    call print_dec
    say " bytes", 13, 10
.tally:
    inc ebx
    mov rcx, [ta+TA_DATA]
    mov rdx, [ta+TA_SIZE]
    mov r8, [ta+TA_NAME]
    call one
    jmp .member
.walked:
    add rsp, 32
    pop rbx
    ret
