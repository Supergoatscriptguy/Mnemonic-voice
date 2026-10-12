; heavy cuts=X.jsonl.gz tar=X.tar out=DIR [shards=64] [threads=0]
; libriheavy -> our shards. reads the cut manifest, finds each chapter's flac in the
; libri-light tar, decodes it (md5 checked), cuts out every utterance, encodes each
; with our encoder and writes DIR\NNNN.shard plus DIR\dev.shard for the held out
; speakers (1 in 100, picked by a hash of the speaker id). each shard is one
; thread's job start to finish, so the output doesn't depend on timing
; uses: data\inflate data\tar data\shardw audio\flac audio\flacenc
default rel
bits 64
%include "lib.inc"
%include "data/inflate.inc"
%include "data/tar.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess

RATE    equ 16000

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
    sub rsp, 336 + UA_SIZE + 8
    ; locals: path +32 (260), cuts left in the chapter +296, UA +304
    mov r12, rcx
    mov ecx, SW_SIZE + FD_SIZE
    call mem_alloc
    mov rbx, rax                ; SW, then the FD
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
    lea rcx, [rsp+32]
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
    lea rcx, [rsp+32]
    mov rdx, [outdir]
    call fmt_str
    mov byte [rax], '\'
    lea rcx, [rax+1]
    lea rdx, [s_dev]
    call fmt_str
    mov byte [rax], 0
.named:
    mov rcx, rbx
    lea rdx, [rsp+32]
    call sw_open
    test eax, eax
    jnz .chapter
    lea rcx, [e_write]
    call fatal
.chapter:
    cmp r13, r14
    jae .finish
    mov rax, [order]
    mov rsi, [rax+r13*8]
    shl rsi, 5
    add rsi, [chaps]            ; CH
    mov rcx, [rsi+CH_DATA]
    mov rdx, [rsi+CH_SIZE]
    lea r8, [rbx+SW_SIZE]
    call flac_decode
    cmp eax, 1
    jne .badch
    cmp dword [rbx+SW_SIZE+FD_MD5], 1
    je .cuts
.badch:
    lock inc qword [badch]
    jmp .nextch
.cuts:
    mov r15, [rsi+CH_CUT]
    imul r15, r15, CU_SIZE
    add r15, [cuts]
    mov eax, [rsi+CH_NCUT]
    mov [rsp+296], rax
.cut:
    cmp qword [rsp+296], 0
    je .chdone
    ; clamped to what the chapter has
    mov rax, [r15+CU_START]
    mov rdx, [rbx+SW_SIZE+FD_N]
    cmp rax, rdx
    jae .nextcut
    mov rcx, [r15+CU_N]
    sub rdx, rax
    cmp rcx, rdx
    cmova rcx, rdx
    test rcx, rcx
    jz .nextcut
    lea rdi, [rsp+304]
    mov [rdi+UA_N], rcx
    lock add [samples], rcx
    mov rdx, [rbx+SW_SIZE+FD_PCM]
    lea rdx, [rdx+rax*2]
    mov [rdi+UA_PCM], rdx
    mov rax, [r15+CU_TEXT]
    mov [rdi+UA_TEXT], rax
    mov eax, [r15+CU_TLEN]
    mov [rdi+UA_TLEN], eax
    mov rax, [r15+CU_NORM]
    mov [rdi+UA_NORM], rax
    mov eax, [r15+CU_NLEN]
    mov [rdi+UA_NLEN], eax
    mov rax, [r15+CU_ID]
    mov [rdi+UA_ID], rax
    mov eax, [r15+CU_ILEN]
    mov [rdi+UA_ILEN], eax
    mov eax, [r15+CU_SPK]
    mov [rdi+UA_SPK], eax
    mov dword [rdi+UA_SRC], SRC_HEAVY
    mov dword [rdi+UA_ESC], 1
    mov rcx, rbx
    mov rdx, rdi
    call sw_add
    lock inc qword [utts]
.nextcut:
    add r15, CU_SIZE
    dec qword [rsp+296]
    jmp .cut
.chdone:
    mov rcx, [rbx+SW_SIZE+FD_PCM]
    call mem_free
.nextch:
    lock inc qword [done]
    call status
    inc r13
    jmp .chapter
.finish:
    mov rcx, rbx
    call sw_close
    lock inc qword [written]
    mov rcx, rbx
    call mem_free
    add rsp, 336 + UA_SIZE + 8
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
