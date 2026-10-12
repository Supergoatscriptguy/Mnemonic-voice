; heavy cuts=X.jsonl.gz tar=X.tar out=DIR [shards=64] [threads=0]
; libriheavy -> our shards. reads the cut manifest, finds each chapter's flac in the
; libri-light tar, decodes it (md5 checked), cuts out every utterance, encodes each
; with our encoder and writes DIR\NNNN.shard plus DIR\dev.shard for the held out
; speakers (1 in 100, picked by a hash of the speaker id). each shard is one
; thread's job start to finish, so the output doesn't depend on timing
; uses: data\inflate data\tar audio\flac audio\flacenc
default rel
bits 64
%include "lib.inc"
%include "data/inflate.inc"
%include "data/tar.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess

RATE    equ 16000
WBUF    equ 8 << 20             ; write buffer per shard

; a cut from the manifest. strings point into the inflated manifest, texts still
; json escaped
CU_REC   equ 0
CU_RLEN  equ 8                  ; u32
CU_SPK   equ 12                 ; u32
CU_START equ 16                 ; first sample
CU_N     equ 24                 ; samples
CU_TEXT  equ 32
CU_TLEN  equ 40                 ; u32
CU_NLEN  equ 44                 ; u32
CU_NORM  equ 48
CU_ID    equ 56
CU_ILEN  equ 64                 ; u32
CU_SIZE  equ 72

; a chapter: its flac in the tar and its run of cuts
CH_DATA  equ 0
CH_SIZE  equ 8
CH_CUT   equ 16                 ; first cut
CH_NCUT  equ 24                 ; u32
CH_DEV   equ 28                 ; u32, held out
CH_SIZEOF equ 32

; tar name hash table entry
HE_NAME  equ 0
HE_LEN   equ 8
HE_DATA  equ 16
HE_SIZE  equ 24
HE_SIZEOF equ 32

; one shard being built
B_FILE   equ 0
B_BUF    equ 8
B_N      equ 16                 ; bytes in the buffer
B_OFF    equ 24                 ; file offset of the buffer's first byte
B_IDX    equ 32
B_NIDX   equ 40
B_SAMP   equ 48
B_PATH   equ 64
B_TMP    equ 320
B_FD     equ 576
B_FE     equ B_FD + FD_SIZE
B_SIZE   equ B_FE + FE_SIZE

section .rdata
k_cuts  db "cuts", 0
k_tar   db "tar", 0
k_out   db "out", 0
k_shard db "shards", 0
k_thr   db "threads", 0
j_id    db '"id": "', 0
j_start db '"start": ', 0
j_dur   db '"duration": ', 0
j_rec   db '"recording_id": "', 0
j_spk   db '"speaker": "', 0
j_texts db '"texts": ["', 0
j_next  db '", "', 0
s_flac  db ".flac", 0
s_shard db ".shard", 0
s_dev   db "dev.shard", 0
s_tmp   db ".tmp", 0
usage   db "usage: heavy cuts=X.jsonl.gz tar=X.tar out=DIR [shards=64] [threads=0]", 13, 10, 0
e_man   db "can't read the manifest", 0
e_gz    db "the manifest's gzip is broken", 0
e_tar   db "can't map the tar", 0
e_write db "a shard write failed", 0
align 8
c_rate  dq 16000.0
c_half  dq 0.5
c_mb    dq 1e-6
c_hour  dq 3600.0

section .bss
alignb 8
inf      resb IF_SIZE
ta       resb TA_SIZEOF
names    resb ARENA_SIZE
manifest resq 1
mlen     resq 1
cuts     resq 1
ncuts    resq 1
chaps    resq 1
nchap    resq 1
order    resq 1                 ; chapter numbers: the training ones, then the held out
ntrain   resq 1
ndev     resq 1
htab     resq 1
hmask    resq 1
missing  resq 1
nshards  resq 1
outdir   resq 1
done     resq 1                 ; chapters
badch    resq 1
utts     resq 1
samples  resq 1
written  resq 1
t0       resq 1
last     resq 1
plock    resd 1

section .text

; rcx = from, rdx = to, r8 = key (zero terminated). rax = just past the key, 0 if
; it isn't there
find:
    push rsi
    push rdi
    mov rsi, rcx
.at:
    cmp rsi, rdx
    jae .no
    mov rdi, rsi
    mov r9, r8
.cmp:
    mov al, [r9]
    test al, al
    jz .yes
    cmp rdi, rdx
    jae .no
    cmp al, [rdi]
    jne .next
    inc rdi
    inc r9
    jmp .cmp
.next:
    inc rsi
    jmp .at
.yes:
    mov rax, rdi
    jmp .out
.no:
    xor eax, eax
.out:
    pop rdi
    pop rsi
    ret

; rcx = just inside a json string. rax = its closing quote
jend:
    mov rax, rcx
.c:
    mov dl, [rax]
    cmp dl, '"'
    je .out
    test dl, dl
    jz .out
    cmp dl, 10
    je .out
    inc rax
    cmp dl, '\'
    jne .c
    inc rax                     ; whatever's escaped
    jmp .c
.out:
    ret

; 4 hex digits at rcx. eax = value
hex4:
    xor eax, eax
    mov r9d, 4
.d:
    movzx edx, byte [rcx]
    inc rcx
    or edx, 0x20                ; lower case letters, digits don't care
    sub edx, '0'
    cmp edx, 9
    jbe .digit
    sub edx, 'a' - '0' - 10
.digit:
    shl eax, 4
    or eax, edx
    dec r9d
    jnz .d
    ret

; rcx = escaped json string, rdx = its end, r8 = out. writes it as utf-8,
; rax = the end of what got written (never longer than the input)
unesc:
    push rbx
    push rsi
    push rdi
    mov rsi, rcx
    mov rbx, rdx
    mov rdi, r8
.c:
    cmp rsi, rbx
    jae .out
    mov al, [rsi]
    inc rsi
    cmp al, '\'
    je .esc
    mov [rdi], al
    inc rdi
    jmp .c
.esc:
    mov al, [rsi]
    inc rsi
    cmp al, 'u'
    je .u
    mov dl, 10
    cmp al, 'n'
    je .put
    mov dl, 9
    cmp al, 't'
    je .put
    mov dl, 13
    cmp al, 'r'
    je .put
    mov dl, 8
    cmp al, 'b'
    je .put
    mov dl, 12
    cmp al, 'f'
    je .put
    mov dl, al                  ; \" \\ \/
.put:
    mov [rdi], dl
    inc rdi
    jmp .c
.u:
    mov rcx, rsi
    call hex4
    add rsi, 4
    cmp eax, 0xd800             ; a high surrogate takes the next \uXXXX with it
    jb .cp
    cmp eax, 0xdbff
    ja .cp
    cmp word [rsi], '\u'        ; nasm packs it low byte first, same as memory
    jne .cp
    mov r10d, eax
    lea rcx, [rsi+2]
    call hex4
    add rsi, 6
    sub r10d, 0xd800
    shl r10d, 10
    sub eax, 0xdc00
    add eax, r10d
    add eax, 0x10000
.cp:
    ; code point -> utf-8
    cmp eax, 0x80
    jae .two
    mov [rdi], al
    inc rdi
    jmp .c
.two:
    cmp eax, 0x800
    jae .three
    mov edx, eax
    shr edx, 6
    or dl, 0xc0
    mov [rdi], dl
    and al, 0x3f
    or al, 0x80
    mov [rdi+1], al
    add rdi, 2
    jmp .c
.three:
    cmp eax, 0x10000
    jae .four
    mov edx, eax
    shr edx, 12
    or dl, 0xe0
    mov [rdi], dl
    mov edx, eax
    shr edx, 6
    and dl, 0x3f
    or dl, 0x80
    mov [rdi+1], dl
    and al, 0x3f
    or al, 0x80
    mov [rdi+2], al
    add rdi, 3
    jmp .c
.four:
    mov edx, eax
    shr edx, 18
    or dl, 0xf0
    mov [rdi], dl
    mov edx, eax
    shr edx, 12
    and dl, 0x3f
    or dl, 0x80
    mov [rdi+1], dl
    mov edx, eax
    shr edx, 6
    and dl, 0x3f
    or dl, 0x80
    mov [rdi+2], dl
    and al, 0x3f
    or al, 0x80
    mov [rdi+3], al
    add rdi, 4
    jmp .c
.out:
    mov rax, rdi
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = bytes, rdx = length. rax = fnv-1a 64
fnv:
    mov rax, 0xcbf29ce484222325
    mov r8, 0x100000001b3
.b:
    test rdx, rdx
    jz .out
    movzx r9d, byte [rcx]
    xor rax, r9
    imul rax, r8
    inc rcx
    dec rdx
    jmp .b
.out:
    ret

; rcx = name, rdx = length. rax = its hash table entry, 0 if it isn't there
lookup:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    mov rdi, rdx
    call fnv
    mov rbx, rax
.probe:
    and rbx, [hmask]
    mov rax, rbx
    shl rax, 5
    add rax, [htab]
    mov rcx, [rax+HE_NAME]
    test rcx, rcx
    jz .no
    cmp [rax+HE_LEN], rdi
    jne .next
    push rsi
    push rdi
    mov rdx, rdi
    mov rdi, rcx
    mov rcx, rdx
    repe cmpsb
    pop rdi
    pop rsi
    je .out
.next:
    inc rbx
    jmp .probe
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = path. inflates the whole manifest into manifest / mlen
load_manifest:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_man]
    call fatal
.mapped:
    mov rsi, rax
    mov rbx, rdx
    mov ecx, [rsi+rbx-4]        ; gzip's length mod 2^32: plenty for small and medium
    add rcx, 64
    call mem_alloc
    mov [manifest], rax
    mov rdi, rax
    lea rcx, [inf]
    mov rdx, rsi
    mov r8, rbx
    mov r9d, 4 << 20
    call gz_open
.run:
    lea rcx, [inf]
    call inflate_run
    mov ebx, eax
    cmp eax, -1
    jne .copy
    lea rcx, [e_gz]
    call fatal
.copy:
    mov rcx, [inf+IF_N]
    sub rcx, [inf+IF_FROM]
    mov rsi, [inf+IF_OUT]
    add rsi, [inf+IF_FROM]
    rep movsb
    cmp ebx, 1
    je .run
    lea rcx, [inf]
    call gz_check
    test eax, eax
    jnz .ok
    lea rcx, [e_gz]
    call fatal
.ok:
    mov rax, rdi
    sub rax, [manifest]
    mov [mlen], rax
    lea rcx, [inf]
    call inflate_free
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; xmm0 = seconds. rax = samples, rounded
tosamples:
    mulsd xmm0, [c_rate]
    cvtsd2si rax, xmm0
    ret

; the manifest's lines -> cuts / ncuts
parse_cuts:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    mov rsi, [manifest]
    mov rcx, [mlen]
    xor ebx, ebx
.count:
    test rcx, rcx
    jz .counted
    cmp byte [rsi+rcx-1], 10
    jne .nc
    inc rbx
.nc:
    dec rcx
    jmp .count
.counted:
    imul rcx, rbx, CU_SIZE
    add rcx, CU_SIZE
    call mem_alloc
    mov [cuts], rax
    mov rdi, rax
    xor r13d, r13d              ; cuts made
    mov r12, [manifest]
    add r12, [mlen]             ; end of it all
.line:
    cmp rsi, r12
    jae .done
    mov rbx, rsi                ; the end of this line
.eol:
    cmp rbx, r12
    jae .gotline
    cmp byte [rbx], 10
    je .gotline
    inc rbx
    jmp .eol
.gotline:
    ; the keys come in this order in every line, so each search goes on from the last
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [j_id]
    call find
    test rax, rax
    jz .skip
    mov [rdi+CU_ID], rax
    mov rcx, rax
    call jend
    sub rax, [rdi+CU_ID]
    mov [rdi+CU_ILEN], eax
    mov rcx, [rdi+CU_ID]
    mov rdx, rbx
    lea r8, [j_start]
    call find
    test rax, rax
    jz .skip
    mov rcx, rax
    call parse_float
    mov rsi, rax
    call tosamples
    mov [rdi+CU_START], rax
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [j_dur]
    call find
    test rax, rax
    jz .skip
    mov rcx, rax
    call parse_float
    mov rsi, rax
    call tosamples
    mov [rdi+CU_N], rax
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [j_rec]
    call find
    test rax, rax
    jz .skip
    mov [rdi+CU_REC], rax
    mov rcx, rax
    call jend
    mov rsi, rax
    sub rax, [rdi+CU_REC]
    mov [rdi+CU_RLEN], eax
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [j_spk]
    call find
    test rax, rax
    jz .skip
    mov rcx, rax
    call parse_int
    mov [rdi+CU_SPK], eax
    mov rcx, rdx
    mov rdx, rbx
    lea r8, [j_texts]
    call find
    test rax, rax
    jz .skip
    mov [rdi+CU_TEXT], rax
    mov rcx, rax
    call jend
    mov rsi, rax
    sub rax, [rdi+CU_TEXT]
    mov [rdi+CU_TLEN], eax
    mov rcx, rsi
    mov rdx, rbx
    lea r8, [j_next]
    call find
    test rax, rax
    jz .skip
    mov [rdi+CU_NORM], rax
    mov rcx, rax
    call jend
    sub rax, [rdi+CU_NORM]
    mov [rdi+CU_NLEN], eax
    add rdi, CU_SIZE
    inc r13
.skip:
    lea rsi, [rbx+1]
    jmp .line
.done:
    mov [ncuts], r13
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = tar path. every .flac member into the hash table, keyed by its name
; without the extension (which is what the manifest calls a recording)
index_tar:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_tar]
    call fatal
.mapped:
    mov rsi, rax
    mov rbx, rdx
    ; a slot for every 64 KB of tar, at least 4096, a power of two
    mov rax, rbx
    shr rax, 16
    mov ecx, 4096
    cmp rax, rcx
    cmovb rax, rcx
    bsr rcx, rax
    inc ecx
    mov eax, 1
    shl rax, cl
    lea rdx, [rax-1]
    mov [hmask], rdx
    mov rcx, rax
    shl rcx, 5
    call mem_alloc
    mov [htab], rax
    lea rcx, [names]
    mov rdx, 1 << 32
    call arena_init
    lea rcx, [ta]
    mov rdx, rsi
    mov r8, rbx
    call tar_open
.member:
    lea rcx, [ta]
    call tar_next
    cmp eax, 1
    jne .done
    mov rsi, [ta+TA_NAME]
    mov rcx, rsi
.end:
    cmp byte [rcx], 0
    je .ended
    inc rcx
    jmp .end
.ended:
    lea r12, [rcx-5]
    cmp r12, rsi
    jb .member
    cmp dword [r12+1], 'flac'
    jne .member
    cmp byte [r12], '.'
    jne .member
    sub r12, rsi                ; name length without .flac
    lea rcx, [names]
    mov rdx, r12
    mov r8d, 1
    call arena_alloc
    mov rdi, rax
    mov rcx, r12
    push rdi
    rep movsb
    pop rdi
    ; insert: probe from its hash to a free slot
    mov rcx, rdi
    mov rdx, r12
    call fnv
.probe:
    and rax, [hmask]
    mov rdx, rax
    shl rdx, 5
    add rdx, [htab]
    cmp qword [rdx+HE_NAME], 0
    je .slot
    inc rax
    jmp .probe
.slot:
    mov [rdx+HE_NAME], rdi
    mov [rdx+HE_LEN], r12
    mov rax, [ta+TA_DATA]
    mov [rdx+HE_DATA], rax
    mov rax, [ta+TA_SIZE]
    mov [rdx+HE_SIZE], rax
    jmp .member
.done:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; runs of cuts with the same recording -> chapters, matched with the tar, and the
; order to build them in: training chapters, then the held out ones
group:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov rcx, [ncuts]
    shl rcx, 5
    add rcx, 64
    call mem_alloc
    mov [chaps], rax
    mov rdi, rax
    mov rcx, [ncuts]
    lea rcx, [rcx*8+64]
    call mem_alloc
    mov [order], rax
    mov rsi, [cuts]
    xor ebx, ebx                ; cut
.chapter:
    cmp rbx, [ncuts]
    jae .grouped
    imul r12, rbx, CU_SIZE
    add r12, rsi                ; its first cut
    mov r13, rbx
.same:
    inc r13
    cmp r13, [ncuts]
    jae .run
    imul rax, r13, CU_SIZE
    add rax, rsi
    mov ecx, [rax+CU_RLEN]
    cmp ecx, [r12+CU_RLEN]
    jne .run
    push rsi
    push rdi
    mov rsi, [rax+CU_REC]
    mov rdi, [r12+CU_REC]
    repe cmpsb
    pop rdi
    pop rsi
    je .same
.run:
    mov rcx, [r12+CU_REC]
    mov edx, [r12+CU_RLEN]
    call lookup
    test rax, rax
    jnz .found
    mov rax, r13
    sub rax, rbx
    add [missing], rax
    jmp .nextch
.found:
    mov rcx, [rax+HE_DATA]
    mov [rdi+CH_DATA], rcx
    mov rcx, [rax+HE_SIZE]
    mov [rdi+CH_SIZE], rcx
    mov [rdi+CH_CUT], rbx
    mov eax, r13d
    sub eax, ebx
    mov [rdi+CH_NCUT], eax
    ; held out: 1 speaker in 100
    lea rcx, [r12+CU_SPK]
    mov edx, 4
    call fnv
    xor edx, edx
    mov ecx, 100
    div rcx
    xor eax, eax
    test edx, edx
    sete al
    mov [rdi+CH_DEV], eax
    add rdi, CH_SIZEOF
    inc qword [nchap]
.nextch:
    mov rbx, r13
    jmp .chapter
.grouped:
    ; training chapters first, then the held out ones
    mov rsi, [chaps]
    mov rdi, [order]
    xor ebx, ebx
.tr:
    cmp rbx, [nchap]
    jae .trd
    imul rax, rbx, CH_SIZEOF
    cmp dword [rsi+rax+CH_DEV], 0
    jne .trn
    mov [rdi], rbx
    add rdi, 8
    inc qword [ntrain]
.trn:
    inc rbx
    jmp .tr
.trd:
    xor ebx, ebx
.dv:
    cmp rbx, [nchap]
    jae .dvd
    imul rax, rbx, CH_SIZEOF
    cmp dword [rsi+rax+CH_DEV], 0
    je .dvn
    mov [rdi], rbx
    add rdi, 8
    inc qword [ndev]
.dvn:
    inc rbx
    jmp .dv
.dvd:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rbx = B. writes out the buffer
flushb:
    push rsi
    sub rsp, 32
    mov rcx, [rbx+B_FILE]
    mov rdx, [rbx+B_BUF]
    mov r8, [rbx+B_N]
    test r8, r8
    jz .out
    call file_write
    test eax, eax
    jnz .ok
    lea rcx, [e_write]
    call fatal
.ok:
    mov rax, [rbx+B_N]
    add [rbx+B_OFF], rax
    mov qword [rbx+B_N], 0
.out:
    add rsp, 32
    pop rsi
    ret

; rbx = B, rcx = bytes it's about to add. flushes first if they won't fit.
; rax = where they go, rdx = their file offset
room:
    sub rsp, 40
    mov rax, [rbx+B_N]
    add rax, rcx
    cmp rax, WBUF
    jbe .fits
    call flushb
.fits:
    mov rax, [rbx+B_BUF]
    add rax, [rbx+B_N]
    mov rdx, [rbx+B_OFF]
    add rdx, [rbx+B_N]
    add rsp, 40
    ret

; the one line status, redrawn by whoever gets there every half second
status:
    push rbx
    sub rsp, 48
    cmp dword [con_tty], 0
    je .ret
    mov rcx, [last]
    call time_since
    comisd xmm0, [c_half]
    jb .ret
    lock bts dword [plock], 0
    jc .ret
    call time_now
    mov [last], rax
    say 13, "  "
    mov rcx, [done]
    call print_dec
    say "/"
    mov rcx, [nchap]
    call print_dec
    say " chapters, "
    mov rcx, [utts]
    call print_dec
    say " utterances, "
    cvtsi2sd xmm0, qword [samples]
    divsd xmm0, [c_rate]
    divsd xmm0, [c_hour]
    mov edx, 1
    call print_fixed
    say " h, "
    mov rcx, [t0]
    call time_since
    cvtsi2sd xmm1, qword [samples]
    divsd xmm1, [c_rate]
    divsd xmm1, xmm0
    movapd xmm0, xmm1
    xor edx, edx
    call print_fixed
    say "x real time", 27, "[K"
    mov dword [plock], 0
.ret:
    add rsp, 48
    pop rbx
    ret

; par_for callback: shards rdx..r8 (index nshards = the held out one)
task:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rdx
    mov rdi, r8
.s:
    cmp rsi, rdi
    jae .out
    mov rcx, rsi
    call build
    inc rsi
    jmp .s
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = shard number. builds it start to finish
build:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov r12, rcx
    mov ecx, B_SIZE
    call mem_alloc
    mov rbx, rax
    mov ecx, WBUF
    call mem_alloc
    mov [rbx+B_BUF], rax
    ; its chapters: a slice of the training ones, or all the held out ones
    cmp r12, [nshards]
    jae .devshard
    mov rax, [ntrain]
    mul r12
    div qword [nshards]
    mov r13, rax                ; first, in the order array
    lea rax, [r12+1]
    mul qword [ntrain]
    div qword [nshards]
    mov r14, rax                ; end
    lea rcx, [rbx+B_PATH]
    mov rdx, [outdir]
    call fmt_str
    mov byte [rax], '\'
    lea rcx, [rax+1]
    mov rdx, r12
    mov r8d, 4
    call fmt_dec0
    mov rcx, rax
    lea rdx, [s_shard]
    call fmt_str
    mov byte [rax], 0
    jmp .named
.devshard:
    mov r13, [ntrain]
    mov r14, r13
    add r14, [ndev]
    lea rcx, [rbx+B_PATH]
    mov rdx, [outdir]
    call fmt_str
    mov byte [rax], '\'
    lea rcx, [rax+1]
    lea rdx, [s_dev]
    call fmt_str
    mov byte [rax], 0
.named:
    lea rcx, [rbx+B_TMP]
    lea rdx, [rbx+B_PATH]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_tmp]
    call fmt_str
    mov byte [rax], 0
    ; room for every cut's index entry
    xor r15d, r15d
    mov rcx, r13
.cnt:
    cmp rcx, r14
    jae .counted
    mov rax, [order]
    mov rax, [rax+rcx*8]
    shl rax, 5
    add rax, [chaps]
    mov edx, [rax+CH_NCUT]
    add r15, rdx
    inc rcx
    jmp .cnt
.counted:
    imul rcx, r15, UT_SIZE
    add rcx, 64
    call mem_alloc
    mov [rbx+B_IDX], rax
    lea rcx, [rbx+B_TMP]
    call file_create
    cmp rax, -1
    jne .opened
    lea rcx, [e_write]
    call fatal
.opened:
    mov [rbx+B_FILE], rax

.chapter:
    cmp r13, r14
    jae .finish
    mov rax, [order]
    mov rsi, [rax+r13*8]
    shl rsi, 5
    add rsi, [chaps]            ; CH
    mov rcx, [rsi+CH_DATA]
    mov rdx, [rsi+CH_SIZE]
    lea r8, [rbx+B_FD]
    call flac_decode
    cmp eax, 1
    jne .badch
    cmp dword [rbx+B_FD+FD_MD5], 1
    je .cuts
.badch:
    lock inc qword [badch]
    jmp .nextch
.cuts:
    mov r15, [rsi+CH_CUT]
    imul r15, r15, CU_SIZE
    add r15, [cuts]
    mov eax, [rsi+CH_NCUT]
    mov [rsp+32], rax           ; cuts left in this chapter
.cut:
    cmp qword [rsp+32], 0
    je .chdone
    ; clamp to what the chapter has
    mov rax, [r15+CU_START]
    mov rdx, [rbx+B_FD+FD_N]
    cmp rax, rdx
    jae .nextcut
    mov rcx, [r15+CU_N]
    sub rdx, rax
    cmp rcx, rdx
    cmova rcx, rdx
    test rcx, rcx
    jz .nextcut
    mov [rsp+40], rcx
    mov rdx, rcx
    mov rcx, [rbx+B_FD+FD_PCM]
    lea rcx, [rcx+rax*2]
    mov r8d, RATE
    lea r9, [rbx+B_FE]
    call flac_encode
    ; the index entry
    mov rdi, [rbx+B_NIDX]
    imul rdi, rdi, UT_SIZE
    add rdi, [rbx+B_IDX]
    mov rcx, [rbx+B_FE+FE_LEN]
    call room
    mov [rdi+UT_FLAC], rdx
    mov rcx, [rbx+B_FE+FE_LEN]
    mov [rdi+UT_FLEN], ecx
    push rsi
    push rdi
    mov rdi, rax
    mov rsi, [rbx+B_FE+FE_DATA]
    rep movsb
    pop rdi
    pop rsi
    mov rax, [rbx+B_FE+FE_LEN]
    add [rbx+B_N], rax
    mov rcx, [rbx+B_FE+FE_DATA]
    call mem_free
    mov rax, [rsp+40]
    mov [rdi+UT_N], eax
    add [rbx+B_SAMP], rax
    lock add [samples], rax
    ; the texts, unescaped, and the id
    mov ecx, [r15+CU_TLEN]
    call room
    mov [rdi+UT_TEXT], rdx
    mov rcx, [r15+CU_TEXT]
    mov edx, [r15+CU_TLEN]
    add rdx, rcx
    mov r8, rax
    mov [rsp+40], rax
    call unesc
    sub rax, [rsp+40]
    mov [rdi+UT_TLEN], eax
    add [rbx+B_N], rax
    mov ecx, [r15+CU_NLEN]
    call room
    mov [rdi+UT_NORM], rdx
    mov rcx, [r15+CU_NORM]
    mov edx, [r15+CU_NLEN]
    add rdx, rcx
    mov r8, rax
    mov [rsp+40], rax
    call unesc
    sub rax, [rsp+40]
    mov [rdi+UT_NLEN], eax
    add [rbx+B_N], rax
    mov ecx, [r15+CU_ILEN]
    call room
    mov [rdi+UT_ID], rdx
    mov ecx, [r15+CU_ILEN]
    mov [rdi+UT_ILEN], ecx
    add [rbx+B_N], rcx
    push rsi
    push rdi
    mov rdi, rax
    mov rsi, [r15+CU_ID]
    rep movsb
    pop rdi
    pop rsi
    mov eax, [r15+CU_SPK]
    mov [rdi+UT_SPK], eax
    mov dword [rdi+UT_SRC], SRC_HEAVY
    inc qword [rbx+B_NIDX]
    lock inc qword [utts]
.nextcut:
    add r15, CU_SIZE
    dec qword [rsp+32]
    jmp .cut
.chdone:
    mov rcx, [rbx+B_FD+FD_PCM]
    call mem_free
.nextch:
    lock inc qword [done]
    call status
    inc r13
    jmp .chapter

.finish:
    ; index, then the trailer
    mov rsi, [rbx+B_NIDX]
    imul rsi, rsi, UT_SIZE
    call flushb
    mov rax, [rbx+B_OFF]
    mov [rsp+40], rax
    mov rcx, [rbx+B_FILE]
    mov rdx, [rbx+B_IDX]
    mov r8, rsi
    test r8, r8
    jz .noidx
    call file_write
    add [rbx+B_OFF], rsi
.noidx:
    mov rdi, [rbx+B_BUF]
    xor eax, eax
    mov ecx, SH_SIZE / 8
    rep stosq
    mov rdi, [rbx+B_BUF]
    mov rax, 'MVSHARD1'
    mov [rdi+SH_MAGIC], rax
    mov rax, [rbx+B_NIDX]
    mov [rdi+SH_COUNT], rax
    mov rax, [rsp+40]
    mov [rdi+SH_INDEX], rax
    mov rax, [rbx+B_SAMP]
    mov [rdi+SH_SAMPLES], rax
    mov dword [rdi+SH_RATE], RATE
    mov qword [rbx+B_N], SH_SIZE
    call flushb
    mov rcx, [rbx+B_FILE]
    call file_flush
    mov rcx, [rbx+B_FILE]
    call file_close
    lea rcx, [rbx+B_TMP]
    lea rdx, [rbx+B_PATH]
    call file_replace
    test eax, eax
    jnz .renamed
    lea rcx, [e_write]
    call fatal
.renamed:
    lock inc qword [written]
    mov rcx, [rbx+B_IDX]
    call mem_free
    mov rcx, [rbx+B_BUF]
    call mem_free
    mov rcx, rbx
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

global start
start:
    push rbx
    push rsi
    sub rsp, 40
    call lib_init
    call cfg_args
    lea rcx, [k_cuts]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov rbx, rax
    lea rcx, [k_tar]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov rsi, rax
    lea rcx, [k_out]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov [outdir], rax
    lea rcx, [k_shard]
    mov edx, 64
    call cfg_int
    mov [nshards], rax

    call time_now
    mov [t0], rax
    mov rcx, rbx
    call load_manifest
    call parse_cuts
    say "manifest: "
    mov rcx, [mlen]
    call print_dec
    say " bytes, "
    mov rcx, [ncuts]
    call print_dec
    say " cuts", 13, 10
    mov rcx, rsi
    call index_tar
    call group
    say "chapters: "
    mov rcx, [nchap]
    call print_dec
    say " in the tar ("
    mov rcx, [ndev]
    call print_dec
    say " held out), cuts with no chapter: "
    mov rcx, [missing]
    call print_dec
    say 13, 10
    mov rcx, [outdir]
    call make_path

    lea rcx, [k_thr]
    xor edx, edx
    call cfg_int
    mov ecx, eax
    call pool_init
    call time_now
    mov [t0], rax
    lea rcx, [task]
    xor edx, edx
    mov r8, [nshards]
    inc r8
    mov r9d, 1
    call par_for
    mov qword [last], 0
    call status
    say 13, 10
    mov rcx, [written]
    call print_dec
    say " shards, chapters that failed their md5: "
    mov rcx, [badch]
    call print_dec
    say 13, 10
    call con_restore
    mov rcx, [badch]
    call ExitProcess
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
