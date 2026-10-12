; cvshard tar=X.tar out=DIR [shards=0] [threads=0]
; common voice -> our shards. scripted speech: validated.tsv's clips, less any clip
; or speaker that's in dev.tsv or test.tsv, go to DIR\NNNN.shard, and dev.tsv and
; test.tsv to DIR\dev.shard and DIR\test.shard. spontaneous speech (an
; ss-corpus-*.tsv): its split column says which, clips with no split stay out.
; each mp3 is decoded, resampled to 16 kHz and encoded with our flac. transcripts
; as written, less their [tags] and <tags>; a clip marked <unclear> stays out.
; shards=0: about 8000 clips a shard. each shard is one thread's job
; uses: data\tar data\shardw audio\mp3 audio\resample audio\flac audio\flacenc
default rel
bits 64
%include "lib.inc"
%include "data/tar.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess

PERSHARD equ 8000
MAXF     equ 64

; an mp3 in the tar
EN_HASH  equ 0                  ; of its file name
EN_DATA  equ 8
EN_LEN   equ 16
EN_SIZE  equ 24

; a clip to write
RW_PATH  equ 0
RW_PLEN  equ 8                  ; u32
RW_SET   equ 12                 ; u32 0 train, 1 dev, 2 test
RW_TEXT  equ 16                 ; cleaned, in the arena
RW_TLEN  equ 24
RW_SPK   equ 28
RW_DATA  equ 32                 ; the mp3, 0 if the tar doesn't have it
RW_SIZE  equ 40
RW_SZ    equ 48

; a worker's things
W_SW     equ 0
W_M3     equ SW_SIZE
W_CLIP   equ W_M3 + M3_SIZE
W_CONV   equ W_CLIP + CL_SIZE
W_UA     equ W_CONV + CL_SIZE
W_PCM    equ W_UA + UA_SIZE     ; int16 samples, grown as needed
W_CAP    equ W_PCM + 8
W_PATH   equ W_CAP + 8
W_SIZE   equ W_PATH + 280

section .rdata
k_tar    db "tar", 0
k_out    db "out", 0
k_shards db "shards", 0
k_thr    db "threads", 0
s_val    db "validated.tsv", 0
s_dev    db "dev.tsv", 0
s_test   db "test.tsv", 0
s_ss     db "ss-corpus-", 0
c_client db "client_id", 0
c_path   db "path", 0
c_file   db "audio_file", 0
c_sent   db "sentence", 0
c_trans  db "transcription", 0
c_split  db "split", 0
v_train  db "train", 0
v_dev    db "dev", 0
v_test   db "test", 0
s_unclr  db "unclear", 0
s_shard  db ".shard", 0
s_devsh  db "dev.shard", 0
s_testsh db "test.shard", 0
usage    db "usage: cvshard tar=X.tar out=DIR [shards=0] [threads=0]", 13, 10, 0
e_tar    db "can't read the tar", 0
e_tsv    db "no validated.tsv / dev.tsv / test.tsv or ss-corpus tsv in the tar", 0
e_cols   db "a tsv is missing a column it needs", 0
e_write  db "can't write a shard", 0
align 8
c_rate   dq 16000.0
c_hour   dq 3600.0
c_half   dq 0.5
align 4
c_scale  dd 32768.0
c_lo     dd -32768.0
c_hi     dd 32767.0

section .bss
alignb 8
ta       resb TA_SIZEOF
texts    resb ARENA_SIZE
ents     resb ARENA_SIZE        ; the mp3s, EN_SIZE each, back to back
entbase  resq 1
htab     resq 1                 ; u32 slots: entry + 1, 0 = free
hmask    resq 1
sset     resq 1                 ; hashes of held out clips and speakers
smask    resq 1
tsvp     resq 4                 ; validated, dev, test, ss-corpus: data, size
tsvn     resq 4
rows     resq 1
nrows    resq 1
order    resq 1                 ; train rows, then dev, then test
nset     resq 3
outdir   resq 1
nshards  resq 1
fldp     resq MAXF              ; the fields of the line just read
fldl     resq MAXF
nfld     resq 1
cpath    resq 1                 ; column numbers in the tsv being read
ctext    resq 1
cspk     resq 1
csplit   resq 1
nomp3    resq 1
dropped  resq 1
nmp3     resq 1
bad      resq 1
done     resq 1
utts     resq 1
samples  resq 1
written  resq 1
t0       resq 1
last     resq 1
plock    resd 1

section .text

; rcx = bytes, rdx = length. rax = fnv-1a 64 (never 0)
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
    test rax, rax
    jnz .r
    inc rax
.r:
    ret

; rcx = file name, rdx = length. rax = its entry, 0 if it isn't in the tar. names
; are matched by their 64-bit hash alone
lookup:
    sub rsp, 40
    call fnv
    mov r10, rax
    mov rcx, rax
.probe:
    and rcx, [hmask]
    mov rdx, [htab]
    mov eax, [rdx+rcx*4]
    test eax, eax
    jz .no
    dec eax
    imul rax, EN_SIZE
    add rax, [entbase]
    cmp [rax+EN_HASH], r10
    je .out
    inc rcx
    jmp .probe
.no:
    xor eax, eax
.out:
    add rsp, 40
    ret

; rax = a hash: into the held out set
sadd:
    mov rcx, rax
.p:
    and rcx, [smask]
    mov rdx, [sset]
    mov r8, [rdx+rcx*8]
    test r8, r8
    jz .put
    cmp r8, rax
    je .out
    inc rcx
    jmp .p
.put:
    mov [rdx+rcx*8], rax
.out:
    ret

; rax = a hash. eax = 1 if it's in the held out set
shas:
    mov rcx, rax
.p:
    and rcx, [smask]
    mov rdx, [sset]
    mov r8, [rdx+rcx*8]
    test r8, r8
    jz .no
    cmp r8, rax
    je .yes
    inc rcx
    jmp .p
.yes:
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; rcx = a, rdx = its length, r8 = b (zero terminated). eax = 1 if they're the same
same:
    xor eax, eax
.c:
    cmp rax, rdx
    jae .end
    mov r9b, [r8+rax]
    test r9b, r9b
    jz .no
    cmp r9b, [rcx+rax]
    jne .no
    inc rax
    jmp .c
.end:
    cmp byte [r8+rax], 0
    jne .no
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; rcx = tar path. every .mp3 in the hash table by its file name; the tsvs noted
index_tar:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 32
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_tar]
    call fatal
.mapped:
    mov rsi, rax
    mov rbx, rdx
    lea rcx, [ents]
    mov rdx, 1 << 34
    call arena_init
    mov rax, [ents+AR_BASE]
    mov [entbase], rax
    lea rcx, [ta]
    mov rdx, rsi
    mov r8, rbx
    call tar_open
.member:
    lea rcx, [ta]
    call tar_next
    cmp eax, 1
    jne .done
    ; its file name: after the last /
    mov rsi, [ta+TA_NAME]
    mov rcx, rsi
.end:
    mov al, [rcx]
    test al, al
    jz .ended
    inc rcx
    cmp al, '/'
    jne .end
    mov rsi, rcx
    jmp .end
.ended:
    mov r12, rcx
    sub r12, rsi                ; its length
    cmp r12, 5
    jb .member
    cmp dword [rcx-4], '.mp3'
    je .mp3
    cmp dword [rcx-4], '.tsv'
    jne .member
    ; one of the tsvs we want?
    xor r13d, r13d
    mov rcx, rsi
    mov rdx, r12
    lea r8, [s_val]
    call same
    test eax, eax
    jnz .tsv
    inc r13d
    mov rcx, rsi
    mov rdx, r12
    lea r8, [s_dev]
    call same
    test eax, eax
    jnz .tsv
    inc r13d
    mov rcx, rsi
    mov rdx, r12
    lea r8, [s_test]
    call same
    test eax, eax
    jnz .tsv
    inc r13d
    mov rcx, rsi
    mov edx, 10
    lea r8, [s_ss]
    call same
    test eax, eax
    jz .member
.tsv:
    lea rax, [tsvp]
    mov rcx, [ta+TA_DATA]
    mov [rax+r13*8], rcx
    lea rax, [tsvn]
    mov rcx, [ta+TA_SIZE]
    mov [rax+r13*8], rcx
    jmp .member
.mp3:
    mov rcx, rsi
    mov rdx, r12
    call fnv
    mov r13, rax
    lea rcx, [ents]
    mov edx, EN_SIZE
    mov r8d, 8
    call arena_alloc
    mov [rax+EN_HASH], r13
    mov rcx, [ta+TA_DATA]
    mov [rax+EN_DATA], rcx
    mov rcx, [ta+TA_SIZE]
    mov [rax+EN_LEN], rcx
    inc qword [nmp3]
    jmp .member
.done:
    ; the table: twice as many slots as mp3s, a power of two
    mov rax, [nmp3]
    add rax, rax
    mov ecx, 4096
    cmp rax, rcx
    cmovb rax, rcx
    bsr rcx, rax
    inc ecx
    mov eax, 1
    shl rax, cl
    lea rdx, [rax-1]
    mov [hmask], rdx
    lea rcx, [rax*4]
    call mem_alloc
    mov [htab], rax
    xor ebx, ebx
.ins:
    cmp rbx, [nmp3]
    jae .indexed
    imul rax, rbx, EN_SIZE
    add rax, [entbase]
    mov rcx, [rax+EN_HASH]
.probe:
    and rcx, [hmask]
    mov rdx, [htab]
    cmp dword [rdx+rcx*4], 0
    je .slot
    inc rcx
    jmp .probe
.slot:
    lea eax, [rbx+1]
    mov [rdx+rcx*4], eax
    inc rbx
    jmp .ins
.indexed:
    add rsp, 32
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = a line, rdx = the tsv's end. its fields into fldp / fldl / nfld.
; rax = the next line
fields:
    xor r8d, r8d
    lea r9, [fldp]
    lea r10, [fldl]
    mov [r9], rcx
    mov rax, rcx
.c:
    cmp rax, rdx
    jae .eol
    mov r11b, [rax]
    cmp r11b, 10
    je .eol
    cmp r11b, 9
    jne .n
    mov r11, rax
    sub r11, [r9+r8*8]
    mov [r10+r8*8], r11
    cmp r8d, MAXF - 1
    jae .n
    inc r8d
    lea r11, [rax+1]
    mov [r9+r8*8], r11
.n:
    inc rax
    jmp .c
.eol:
    mov r11, rax
    cmp r11, [r9+r8*8]
    jbe .last
    cmp byte [r11-1], 13
    jne .last
    dec r11
.last:
    sub r11, [r9+r8*8]
    mov [r10+r8*8], r11
    inc r8d
    mov [nfld], r8
    cmp rax, rdx
    jae .out
    inc rax
.out:
    ret

; rcx = a column's name: eax = its number in the fields just read, -1 if not there
colof:
    push rbx
    push rsi
    sub rsp, 40
    mov rsi, rcx
    xor ebx, ebx
.c:
    cmp rbx, [nfld]
    jae .no
    lea rax, [fldp]
    mov rcx, [rax+rbx*8]
    lea rax, [fldl]
    mov rdx, [rax+rbx*8]
    mov r8, rsi
    call same
    test eax, eax
    jnz .yes
    inc ebx
    jmp .c
.yes:
    mov eax, ebx
    jmp .out
.no:
    mov eax, -1
.out:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = text, rdx = its length. a copy in the arena without [tags] and <tags> and
; with single spaces: rax = it, rdx = its length, 0 if nothing's left, -1 if it
; has <unclear> or [unclear]
clean:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rcx
    mov rbx, rdx
    ; an unclear tag anywhere?
    xor ecx, ecx
.u:
    lea rax, [rcx+8]
    cmp rax, rbx
    ja .alloc
    mov al, [rsi+rcx]
    cmp al, '<'
    je .ut
    cmp al, '['
    jne .un
.ut:
    xor edx, edx
    lea r9, [rsi+rcx+1]
.uc:
    lea rax, [s_unclr]
    mov al, [rax+rdx]
    test al, al
    jz .unclear
    mov r8b, [r9+rdx]
    or r8b, 0x20
    cmp al, r8b
    jne .un
    inc edx
    jmp .uc
.un:
    inc rcx
    jmp .u
.unclear:
    mov rdx, -1
    jmp .out
.alloc:
    lea rcx, [texts]
    lea rdx, [rbx+1]
    mov r8d, 1
    call arena_alloc
    mov rdi, rax
    mov r12, rax
    xor ecx, ecx
    mov r8d, 1                  ; the last thing written was a space (or nothing)
.c:
    cmp rcx, rbx
    jae .end
    mov al, [rsi+rcx]
    cmp al, '['
    je .tag
    cmp al, '<'
    je .tag
    cmp al, ' '
    je .sp
    cmp al, 9
    je .sp
    mov [rdi], al
    inc rdi
    xor r8d, r8d
    inc rcx
    jmp .c
.sp:
    inc rcx
    test r8d, r8d
    jnz .c
    mov byte [rdi], ' '
    inc rdi
    mov r8d, 1
    jmp .c
.tag:
    mov dl, ']'
    cmp al, '['
    je .close
    mov dl, '>'
.close:
    inc rcx
    cmp rcx, rbx
    jae .end
    cmp [rsi+rcx], dl
    jne .close
    inc rcx
    jmp .c
.end:
    ; no space at the end, nor before , . ? ! that a tag left behind
    cmp rdi, r12
    je .len
    cmp byte [rdi-1], ' '
    jne .len
    dec rdi
.len:
    mov rax, r12
    mov rdx, rdi
    sub rdx, r12
    ; a tag before punctuation leaves "word ," : close that up
    xor ecx, ecx
    mov r8, r12
.f:
    cmp rcx, rdx
    jae .fd
    mov r9b, [r12+rcx]
    cmp r9b, ' '
    jne .keep
    lea r10, [rcx+1]
    cmp r10, rdx
    jae .keep
    mov r10b, [r12+rcx+1]
    cmp r10b, ','
    je .skipsp
    cmp r10b, '.'
    je .skipsp
    cmp r10b, '?'
    je .skipsp
    cmp r10b, '!'
    je .skipsp
    cmp r10b, ';'
    je .skipsp
.keep:
    mov [r8], r9b
    inc r8
.skipsp:
    inc rcx
    jmp .f
.fd:
    mov rdx, r8
    sub rdx, r12
    ; and no space at the start
    cmp rdx, 0
    je .out
    cmp byte [r12], ' '
    jne .out
    inc rax
    dec rdx
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = set: the line in fldp/fldl as a row (columns cpath, ctext, cspk)
addrow:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov ebx, ecx
    mov rax, [cpath]
    cmp rax, [nfld]
    jae .out
    mov rax, [ctext]
    cmp rax, [nfld]
    jae .out
    mov rax, [cspk]
    cmp rax, [nfld]
    jae .out
    mov rdi, [nrows]
    imul rdi, RW_SZ
    add rdi, [rows]
    ; the transcript
    mov rax, [ctext]
    lea rcx, [fldp]
    mov rcx, [rcx+rax*8]
    lea rdx, [fldl]
    mov rdx, [rdx+rax*8]
    test rdx, rdx
    jz .out
    call clean
    test rdx, rdx
    jg .text
    inc qword [dropped]
    jmp .out
.text:
    mov [rdi+RW_TEXT], rax
    mov [rdi+RW_TLEN], edx
    mov [rdi+RW_SET], ebx
    ; the clip
    mov rax, [cpath]
    lea rcx, [fldp]
    mov rcx, [rcx+rax*8]
    lea rdx, [fldl]
    mov rdx, [rdx+rax*8]
    mov [rdi+RW_PATH], rcx
    mov [rdi+RW_PLEN], edx
    call lookup
    test rax, rax
    jnz .found
    inc qword [nomp3]
    jmp .out
.found:
    mov rcx, [rax+EN_DATA]
    mov [rdi+RW_DATA], rcx
    mov rcx, [rax+EN_LEN]
    mov [rdi+RW_SIZE], rcx
    ; the speaker, as a hash of the client id
    mov rax, [cspk]
    lea rcx, [fldp]
    mov rcx, [rcx+rax*8]
    lea rdx, [fldl]
    mov rdx, [rdx+rax*8]
    call fnv
    mov [rdi+RW_SPK], eax
    lea rax, [nset]
    inc qword [rax+rbx*8]
    inc qword [nrows]
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; edx = which tsv (0-3): its header's columns into cpath, ctext, cspk, csplit.
; rax = its first row, rdx = its end
header:
    push rbx
    push rsi
    sub rsp, 40
    lea rax, [tsvp]
    mov rcx, [rax+rdx*8]
    lea rax, [tsvn]
    mov rsi, [rax+rdx*8]
    add rsi, rcx
    mov rdx, rsi
    call fields
    mov rbx, rax
    lea rcx, [c_path]
    call colof
    cmp eax, -1
    jne .p
    lea rcx, [c_file]
    call colof
.p:
    movsxd rax, eax
    mov [cpath], rax
    lea rcx, [c_sent]
    call colof
    cmp eax, -1
    jne .t
    lea rcx, [c_trans]
    call colof
.t:
    movsxd rax, eax
    mov [ctext], rax
    lea rcx, [c_client]
    call colof
    movsxd rax, eax
    mov [cspk], rax
    lea rcx, [c_split]
    call colof
    movsxd rax, eax
    mov [csplit], rax
    mov rax, [cpath]
    or rax, [ctext]
    or rax, [cspk]
    jns .ok
    lea rcx, [e_cols]
    call fatal
.ok:
    mov rax, rbx
    mov rdx, rsi
    add rsp, 40
    pop rsi
    pop rbx
    ret

; eax = the field's hash (column rax of the line just read)
fieldhash:
    lea rcx, [fldp]
    mov rcx, [rcx+rax*8]
    lea rdx, [fldl]
    mov rdx, [rdx+rax*8]
    jmp fnv

; edx = which tsv, ecx = the set its rows go to. scripted dev and test: their clips
; and speakers go into the held out set too
readheld:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov ebx, ecx
    call header
    mov rsi, rax
    mov rdi, rdx
.l:
    cmp rsi, rdi
    jae .out
    mov rcx, rsi
    mov rdx, rdi
    call fields
    mov rsi, rax
    mov rax, [cspk]
    cmp rax, [nfld]
    jae .l
    mov rax, [cpath]
    cmp rax, [nfld]
    jae .l
    call fieldhash
    call sadd
    mov rax, [cspk]
    call fieldhash
    call sadd
    mov ecx, ebx
    call addrow
    jmp .l
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; validated.tsv: training rows unless held out
readtrain:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    xor edx, edx
    call header
    mov rsi, rax
    mov rdi, rdx
.l:
    cmp rsi, rdi
    jae .out
    mov rcx, rsi
    mov rdx, rdi
    call fields
    mov rsi, rax
    mov rax, [cspk]
    cmp rax, [nfld]
    jae .l
    mov rax, [cpath]
    cmp rax, [nfld]
    jae .l
    call fieldhash
    call shas
    test eax, eax
    jnz .l
    mov rax, [cspk]
    call fieldhash
    call shas
    test eax, eax
    jnz .l
    xor ecx, ecx
    call addrow
    jmp .l
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; the spontaneous tsv: its split column picks the set
readspont:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    mov edx, 3
    call header
    mov rsi, rax
    mov rdi, rdx
    cmp qword [csplit], 0
    jge .l
    lea rcx, [e_cols]
    call fatal
.l:
    cmp rsi, rdi
    jae .out
    mov rcx, rsi
    mov rdx, rdi
    call fields
    mov rsi, rax
    mov rax, [csplit]
    cmp rax, [nfld]
    jae .l
    lea rcx, [fldp]
    mov rbx, [rcx+rax*8]
    lea rcx, [fldl]
    mov rax, [rcx+rax*8]
    mov [rsp+32], rax
    mov rcx, rbx
    mov rdx, rax
    lea r8, [v_train]
    call same
    xor ecx, ecx
    test eax, eax
    jnz .add
    mov rcx, rbx
    mov rdx, [rsp+32]
    lea r8, [v_dev]
    call same
    mov ecx, 1
    test eax, eax
    jnz .add
    mov rcx, rbx
    mov rdx, [rsp+32]
    lea r8, [v_test]
    call same
    mov ecx, 2
    test eax, eax
    jz .l
.add:
    call addrow
    jmp .l
.out:
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = tsv bytes, rdx = size. rax = its lines
lines:
    xor eax, eax
    xor r8d, r8d
.c:
    cmp r8, rdx
    jae .out
    cmp byte [rcx+r8], 10
    jne .n
    inc rax
.n:
    inc r8
    jmp .c
.out:
    inc rax
    ret

; the order: train rows, then dev, then test
arrange:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rcx, [nrows]
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov [order], rax
    mov rdi, rax
    xor esi, esi                ; set
.set:
    xor ebx, ebx
.r:
    cmp rbx, [nrows]
    jae .nset
    imul rax, rbx, RW_SZ
    add rax, [rows]
    cmp [rax+RW_SET], esi
    jne .n
    mov [rdi], ebx
    add rdi, 4
.n:
    inc rbx
    jmp .r
.nset:
    inc esi
    cmp esi, 3
    jb .set
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
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
    mov rcx, [nrows]
    call print_dec
    say " clips, "
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

; par_for callback: jobs rdx..r8 (nshards is dev, nshards + 1 test)
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

; rcx = job. one shard, start to finish
build:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov r12, rcx
    ; its rows: a slice of training, or dev, or test
    cmp r12, [nshards]
    jae .held
    mov rax, [nset]
    mul r12
    div qword [nshards]
    mov r13, rax
    lea rax, [r12+1]
    mul qword [nset]
    div qword [nshards]
    mov r14, rax
    jmp .rows
.held:
    mov r13, [nset]
    mov r14, r13
    add r14, [nset+8]
    cmp r12, [nshards]
    je .rows
    mov r13, r14
    add r14, [nset+16]
.rows:
    cmp r13, r14
    jae .ret                    ; nothing for it
    mov ecx, W_SIZE
    call mem_alloc
    mov rbx, rax
    lea rcx, [rbx+W_PATH]
    mov rdx, [outdir]
    call fmt_str
    mov byte [rax], '\'
    lea rcx, [rax+1]
    cmp r12, [nshards]
    jae .hname
    mov rdx, r12
    mov r8d, 4
    call fmt_dec0
    mov rcx, rax
    lea rdx, [s_shard]
    call fmt_str
    jmp .named
.hname:
    lea rdx, [s_devsh]
    je .hn
    lea rdx, [s_testsh]
.hn:
    call fmt_str
.named:
    mov byte [rax], 0
    lea rcx, [rbx+W_SW]
    lea rdx, [rbx+W_PATH]
    call sw_open
    test eax, eax
    jnz .clip
    lea rcx, [e_write]
    call fatal
.clip:
    cmp r13, r14
    jae .finish
    mov rax, [order]
    mov esi, [rax+r13*4]
    imul rsi, RW_SZ
    add rsi, [rows]
    mov rcx, [rsi+RW_DATA]
    mov rdx, [rsi+RW_SIZE]
    lea r8, [rbx+W_CLIP]
    lea r9, [rbx+W_M3]
    call mp3_decode
    test eax, eax
    jnz .decoded
    lock inc qword [bad]
    jmp .next
.decoded:
    lea rdi, [rbx+W_CLIP]
    cmp dword [rdi+CL_RATE], 16000
    je .int16
    mov rcx, rdi
    mov edx, 16000
    lea r8, [rbx+W_CONV]
    call clip_resample
    mov rcx, [rbx+W_CLIP+CL_DATA]
    call mem_free
    lea rdi, [rbx+W_CONV]
.int16:
    ; to 16 bits, clipped (a decoder's output can go past full scale)
    mov r15, [rdi+CL_FRAMES]
    cmp r15, [rbx+W_CAP]
    jbe .room
    mov rcx, [rbx+W_PCM]
    test rcx, rcx
    jz .grow
    call mem_free
.grow:
    lea rcx, [r15+r15+65536]
    mov [rbx+W_CAP], rcx
    shr qword [rbx+W_CAP], 1
    call mem_alloc
    mov [rbx+W_PCM], rax
.room:
    mov rax, [rbx+W_PCM]
    mov rdx, [rdi+CL_DATA]
    movss xmm1, [c_scale]
    movss xmm2, [c_lo]
    movss xmm3, [c_hi]
    xor r8d, r8d
.s:
    cmp r8, r15
    jae .converted
    movss xmm0, [rdx+r8*4]
    mulss xmm0, xmm1
    maxss xmm0, xmm2
    minss xmm0, xmm3
    cvtss2si r9d, xmm0
    mov [rax+r8*2], r9w
    inc r8
    jmp .s
.converted:
    mov rcx, [rdi+CL_DATA]
    call mem_free
    lea rdi, [rbx+W_UA]
    mov rax, [rbx+W_PCM]
    mov [rdi+UA_PCM], rax
    mov [rdi+UA_N], r15
    mov rax, [rsi+RW_TEXT]
    mov [rdi+UA_TEXT], rax
    mov eax, [rsi+RW_TLEN]
    mov [rdi+UA_TLEN], eax
    xor eax, eax
    mov [rdi+UA_NORM], rax
    mov [rdi+UA_NLEN], eax
    mov rax, [rsi+RW_PATH]
    mov [rdi+UA_ID], rax
    mov eax, [rsi+RW_PLEN]
    mov [rdi+UA_ILEN], eax
    mov eax, [rsi+RW_SPK]
    mov [rdi+UA_SPK], eax
    mov dword [rdi+UA_SRC], SRC_CV
    mov dword [rdi+UA_ESC], 0
    lea rcx, [rbx+W_SW]
    mov rdx, rdi
    call sw_add
    lock inc qword [utts]
    lock add [samples], r15
.next:
    lock inc qword [done]
    call status
    inc r13
    jmp .clip
.finish:
    lea rcx, [rbx+W_SW]
    call sw_close
    lock inc qword [written]
    mov rcx, [rbx+W_PCM]
    test rcx, rcx
    jz .wf
    call mem_free
.wf:
    mov rcx, rbx
    call mem_free
.ret:
    add rsp, 32
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
    lea rcx, [k_tar]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov rbx, rax
    lea rcx, [k_out]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov [outdir], rax
    call time_now
    mov [t0], rax
    mov rcx, rbx
    call index_tar
    say "tar: "
    mov rcx, [nmp3]
    call print_dec
    say " mp3s", 13, 10
    ; rows: as many as the tsvs have lines, at most
    xor esi, esi
    xor ebx, ebx
.cnt:
    lea rax, [tsvp]
    mov rcx, [rax+rbx*8]
    lea rax, [tsvn]
    mov rdx, [rax+rbx*8]
    call lines
    add rsi, rax
    inc ebx
    cmp ebx, 4
    jb .cnt
    imul rcx, rsi, RW_SZ
    add rcx, 64
    call mem_alloc
    mov [rows], rax
    lea rcx, [texts]
    mov rdx, 1 << 34
    call arena_init
    ; the held out set: two slots a line of dev + test, a power of two
    mov rax, [tsvn+8]
    add rax, [tsvn+16]
    shr rax, 4
    mov ecx, 4096
    cmp rax, rcx
    cmovb rax, rcx
    bsr rcx, rax
    add ecx, 2
    mov eax, 1
    shl rax, cl
    lea rdx, [rax-1]
    mov [smask], rdx
    lea rcx, [rax*8]
    call mem_alloc
    mov [sset], rax
    cmp qword [tsvp+24], 0
    jne .spont
    cmp qword [tsvp], 0
    je .notsv
    cmp qword [tsvp+8], 0
    je .notsv
    cmp qword [tsvp+16], 0
    je .notsv
    mov edx, 1
    mov ecx, 1
    call readheld
    mov edx, 2
    mov ecx, 2
    call readheld
    call readtrain
    jmp .rowsdone
.spont:
    call readspont
    jmp .rowsdone
.notsv:
    lea rcx, [e_tsv]
    call fatal
.rowsdone:
    call arrange
    say "clips: "
    mov rcx, [nset]
    call print_dec
    say " train, "
    mov rcx, [nset+8]
    call print_dec
    say " dev, "
    mov rcx, [nset+16]
    call print_dec
    say " test. left out: "
    mov rcx, [dropped]
    call print_dec
    say " unclear or empty, "
    mov rcx, [nomp3]
    call print_dec
    say " not in the tar", 13, 10
    lea rcx, [k_shards]
    xor edx, edx
    call cfg_int
    test rax, rax
    jnz .ns
    mov rax, [nset]
    add rax, PERSHARD - 1
    xor edx, edx
    mov ecx, PERSHARD
    div rcx
.ns:
    test rax, rax
    jnz .ns1
    mov eax, 1
.ns1:
    mov [nshards], rax
    mov rcx, [outdir]
    call make_path
    call mp3_init
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
    add r8, 2
    mov r9d, 1
    call par_for
    mov qword [last], 0
    call status
    say 13, 10
    mov rcx, [written]
    call print_dec
    say " shards, "
    mov rcx, [utts]
    call print_dec
    say " clips, mp3s we couldn't decode: "
    mov rcx, [bad]
    call print_dec
    say 13, 10
    call con_restore
    mov rcx, [bad]
    call ExitProcess
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
