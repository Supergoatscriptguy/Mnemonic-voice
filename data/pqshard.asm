; pqshard src=vox|ls out=DIR [threads=4] FILE.parquet...
; hugging face speech parquet -> our shards, one shard per file (DIR\<name>.shard).
;   vox: voxpopuli. raw_text (cased, punctuated), normalized_text kept for
;        reference, speaker_id (a string), audio_id. rows that aren't
;        is_gold_transcript get SRC_NOTGOLD on their source
;   ls:  librispeech. text (upper case), the int64 speaker_id (checked against
;        the id's first part), id
; audio can be flac or wav; wav gets resampled to 16 kHz and mixed to mono if it
; has to be, then everything is re-encoded with our encoder. a row group of
; voxpopuli is ~570 MB of audio in memory, so threads default to 4
; uses: data\parquet data\snappy data\zstd data\shardw audio\flac audio\flacenc audio\wav audio\resample
default rel
bits 64
%include "lib.inc"
%include "data/parquet.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess

MAXF    equ 256
NCOL    equ 6                   ; audio, text, norm, speaker, id, gold

; columns, in the order of the name lists below
C_AUDIO equ 0
C_TEXT  equ 1
C_NORM  equ 2
C_SPK   equ 3
C_ID    equ 4
C_GOLD  equ 5

; one worker's things
W_PQ    equ 0
W_AR    equ W_PQ + PQ_SIZEOF
W_COLS  equ W_AR + ARENA_SIZE
W_CIDX  equ W_COLS + COL_SIZE * NCOL     ; leaf index per column, -1 if not used
W_SW    equ W_CIDX + 8 * NCOL
W_FD    equ W_SW + SW_SIZE
W_CLIP  equ W_FD + FD_SIZE
W_CONV  equ W_CLIP + CL_SIZE
W_UA    equ W_CONV + CL_SIZE
W_PATH  equ W_UA + UA_SIZE
W_SIZE  equ W_PATH + 1024

section .rdata
k_src   db "src", 0
k_out   db "out", 0
k_thr   db "threads", 0
c_audio db "audio.bytes", 0
v_text  db "raw_text", 0
v_norm  db "normalized_text", 0
v_spk   db "speaker_id", 0
v_id    db "audio_id", 0
v_gold  db "is_gold_transcript", 0
l_text  db "text", 0
l_id    db "id", 0
s_shard db ".shard", 0
usage   db "usage: pqshard src=vox|ls out=DIR [threads=4] FILE.parquet...", 13, 10, 0
e_col   db "a column this source needs isn't there", 0
e_write db "can't create a shard", 0
align 8
c_rate  dq 16000.0
c_hour  dq 3600.0
c_scale dd 32768.0
c_lo    dd -32768.0
c_hi    dd 32767.0

section .bss
alignb 8
files   resq MAXF
nfiles  resq 1
isvox   resq 1
outdir  resq 1
colname resq NCOL               ; names this source uses (0 = none)
utts    resq 1
samples resq 1
notgold resq 1
spkbad  resq 1
failed  resq 1
skipped resq 1

section .text

; rcx = column name or 0, rdx = W, r8 = which. finds the leaf, or -1
leaf:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rdx
    mov rsi, r8
    mov qword [rbx+W_CIDX+rsi*8], -1
    test rcx, rcx
    jz .out
    mov rdx, rcx
    lea rcx, [rbx+W_PQ]
    call pq_leaf
    cmp rax, -1
    jne .found
    lea rcx, [e_col]
    call fatal
.found:
    mov [rbx+W_CIDX+rsi*8], rax
.out:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rbx = W, ecx = column, rdx = row. rax = value ptr, rdx = length (0, 0 if unused)
value:
    xor eax, eax
    cmp qword [rbx+W_CIDX+rcx*8], -1
    je .none
    imul rcx, rcx, COL_SIZE
    mov rax, [rbx+W_COLS+rcx+COL_VALS]
    shl rdx, 4
    mov r8, [rax+rdx+8]
    mov rax, [rax+rdx]
    mov rdx, r8
    ret
.none:
    xor edx, edx
    ret

; rbx = W, rsi = the audio blob, rdi = its length. W_UA's pcm and n get the 16 kHz
; mono int16 samples (mem_alloc'd, the caller frees UA_PCM). eax = 1, 0 if unusable
audio:
    push r12
    sub rsp, 32
    cmp rdi, 4
    jb .no
    cmp dword [rsi], 'fLaC'
    jne .wav
    mov rcx, rsi
    mov rdx, rdi
    lea r8, [rbx+W_FD]
    call flac_decode
    cmp eax, 1
    jne .no
    cmp dword [rbx+W_FD+FD_MD5], 0
    je .flacbad
    cmp dword [rbx+W_FD+FI_RATE], 16000
    jne .flacbad
    cmp dword [rbx+W_FD+FI_CH], 1
    jne .flacbad
    mov rax, [rbx+W_FD+FD_PCM]
    mov [rbx+W_UA+UA_PCM], rax
    mov rax, [rbx+W_FD+FD_N]
    mov [rbx+W_UA+UA_N], rax
    mov eax, 1
    jmp .out
.flacbad:
    mov rcx, [rbx+W_FD+FD_PCM]
    call mem_free
    jmp .no
.wav:
    mov rcx, rsi
    mov rdx, rdi
    lea r8, [rbx+W_CLIP]
    call wav_parse
    test eax, eax
    jz .no
    lea rcx, [rbx+W_CLIP]
    call clip_mono
    lea r12, [rbx+W_CLIP]
    cmp dword [rbx+W_CLIP+CL_RATE], 16000
    je .tointeger
    lea rcx, [rbx+W_CLIP]
    mov edx, 16000
    lea r8, [rbx+W_CONV]
    call clip_resample
    mov rcx, [rbx+W_CLIP+CL_DATA]
    call mem_free
    lea r12, [rbx+W_CONV]
.tointeger:
    ; round and clip to int16, the same way wav_write does
    mov rcx, [r12+CL_FRAMES]
    mov [rbx+W_UA+UA_N], rcx
    lea rcx, [rcx*2+64]
    call mem_alloc
    mov [rbx+W_UA+UA_PCM], rax
    mov rdx, [r12+CL_DATA]
    mov rcx, [r12+CL_FRAMES]
    movss xmm1, [c_scale]
    movss xmm2, [c_lo]
    movss xmm3, [c_hi]
    xor r8d, r8d
.s:
    cmp r8, rcx
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
    mov rcx, [r12+CL_DATA]
    call mem_free
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 32
    pop r12
    ret

; rcx = file number, rdx = W. that file -> its shard
onefile:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov rbx, rdx
    mov r12, rcx
    lea rax, [files]
    mov rsi, [rax+r12*8]
    ; DIR\<file name without .parquet>.shard
    lea rcx, [rbx+W_PATH]
    mov rdx, [outdir]
    call fmt_str
    mov byte [rax], 92
    lea rdi, [rax+1]
    mov rcx, rsi                ; the name starts after the last slash
    mov rdx, rsi
.sl:
    mov al, [rcx]
    test al, al
    jz .last
    inc rcx
    cmp al, 92
    je .cut
    cmp al, '/'
    jne .sl
.cut:
    mov rdx, rcx
    jmp .sl
.last:
    ; up to its last dot (test.clean.parquet -> test.clean)
    mov r8, rdx
    xor r9d, r9d
.dot:
    mov al, [r8]
    test al, al
    jz .dotted
    cmp al, '.'
    jne .nd
    mov r9, r8
.nd:
    inc r8
    jmp .dot
.dotted:
    test r9, r9
    cmovz r9, r8
.base:
    cmp rdx, r9
    jae .ext
    mov al, [rdx]
    mov [rdi], al
    inc rdi
    inc rdx
    jmp .base
.ext:
    mov rcx, rdi
    lea rdx, [s_shard]
    call fmt_str
    mov byte [rax], 0
    ; open it all
    lea rcx, [rbx+W_PQ]
    mov rdx, rsi
    call pq_open
    test eax, eax
    jnz .opened
    lock inc qword [failed]
    jmp .ret
.opened:
    xor r13d, r13d
.cols:
    lea rax, [colname]
    mov rcx, [rax+r13*8]
    mov rdx, rbx
    mov r8, r13
    call leaf
    inc r13d
    cmp r13d, NCOL
    jb .cols
    lea rcx, [rbx+W_SW]
    lea rdx, [rbx+W_PATH]
    call sw_open
    test eax, eax
    jnz .rgs
    lea rcx, [e_write]
    call fatal
.rgs:
    xor r13d, r13d              ; row group
.rg:
    cmp r13, [rbx+W_PQ+PQ_NRG]
    jae .close
    ; every column this source uses, checked to have a value in every row
    imul rax, r13, RG_SIZE
    add rax, [rbx+W_PQ+PQ_RG]
    mov r15, [rax+RG_ROWS]
    xor r14d, r14d
.chunk:
    cmp qword [rbx+W_CIDX+r14*8], -1
    je .nextcol
    imul rdx, r14, COL_SIZE
    lea rdx, [rbx+W_COLS+rdx]
    lea rax, [rbx+W_AR]
    mov [rdx+COL_ARENA], rax
    lea rcx, [rbx+W_PQ]
    mov r8, r13
    mov r9, [rbx+W_CIDX+r14*8]
    call pq_chunk
    imul rdx, r14, COL_SIZE
    cmp [rbx+W_COLS+rdx+COL_NV], r15
    je .nextcol
    lock add [skipped], r15     ; nulls somewhere: not worth handling
    jmp .rgdone
.nextcol:
    inc r14d
    cmp r14d, NCOL
    jb .chunk
    xor r14d, r14d              ; row
.row:
    cmp r14, r15
    jae .rgdone
    mov ecx, C_AUDIO
    mov rdx, r14
    call value
    mov rsi, rax
    mov rdi, rdx
    call audio
    test eax, eax
    jnz .usable
    lock inc qword [failed]
    jmp .nextrow
.usable:
    mov ecx, C_TEXT
    mov rdx, r14
    call value
    mov [rbx+W_UA+UA_TEXT], rax
    mov [rbx+W_UA+UA_TLEN], edx
    mov ecx, C_NORM
    mov rdx, r14
    call value
    mov [rbx+W_UA+UA_NORM], rax
    mov [rbx+W_UA+UA_NLEN], edx
    mov ecx, C_ID
    mov rdx, r14
    call value
    mov [rbx+W_UA+UA_ID], rax
    mov [rbx+W_UA+UA_ILEN], edx
    mov dword [rbx+W_UA+UA_ESC], 0
    mov ecx, C_SPK
    mov rdx, r14
    call value
    cmp qword [isvox], 0
    je .lsspk
    ; voxpopuli: a number in a string
    mov rcx, rax
    call parse_int
    mov [rbx+W_UA+UA_SPK], eax
    mov dword [rbx+W_UA+UA_SRC], SRC_VOXPOP
    mov ecx, C_GOLD
    mov rdx, r14
    call value
    cmp byte [rax], 0
    jne .add
    or dword [rbx+W_UA+UA_SRC], SRC_NOTGOLD
    lock inc qword [notgold]
    jmp .add
.lsspk:
    ; librispeech: int64, and the id starts with the same number
    mov rax, [rax]
    mov [rbx+W_UA+UA_SPK], eax
    mov dword [rbx+W_UA+UA_SRC], SRC_LS
    mov rcx, [rbx+W_UA+UA_ID]
    call parse_int
    cmp eax, [rbx+W_UA+UA_SPK]
    je .add
    lock inc qword [spkbad]
.add:
    lea rcx, [rbx+W_SW]
    lea rdx, [rbx+W_UA]
    call sw_add
    mov rax, [rbx+W_UA+UA_N]
    lock add [samples], rax
    lock inc qword [utts]
    mov rcx, [rbx+W_UA+UA_PCM]
    call mem_free
.nextrow:
    inc r14
    jmp .row
.rgdone:
    lea rcx, [rbx+W_AR]
    call arena_reset
    inc r13
    jmp .rg
.close:
    lea rcx, [rbx+W_SW]
    call sw_close
    lea rcx, [rbx+W_PQ]
    call pq_close
.ret:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; par_for callback: files rdx..r8. each thread keeps one W
task:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rdx
    mov rdi, r8
    mov rcx, W_SIZE
    call mem_alloc
    mov rbx, rax
    lea rcx, [rbx+W_AR]
    mov rdx, 1 << 34
    call arena_init
    mov ecx, ZSTD_CTX
    call mem_alloc
    mov r12, rax
    mov rcx, rax
    call zstd_init
    xor ecx, ecx
.z:
    imul rax, rcx, COL_SIZE
    mov [rbx+W_COLS+rax+COL_ZSTD], r12
    inc ecx
    cmp ecx, NCOL
    jb .z
.f:
    cmp rsi, rdi
    jae .out
    ; a clean writer each time
    push rdi
    lea rdi, [rbx+W_SW]
    xor eax, eax
    mov ecx, SW_SIZE
    rep stosb
    pop rdi
    mov rcx, rsi
    mov rdx, rbx
    call onefile
    inc rsi
    jmp .f
.out:
    add rsp, 40
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
    ; the files are the args without '=' (cfg_args cuts the others up)
    mov ebx, 1
.arg:
    cmp rbx, [argc]
    jae .args
    lea rax, [argv]
    mov rsi, [rax+rbx*8]
    mov rcx, rsi
.eq:
    mov al, [rcx]
    test al, al
    jz .file
    cmp al, '='
    je .nextarg
    inc rcx
    jmp .eq
.file:
    mov rax, [nfiles]
    cmp rax, MAXF
    jae .nextarg
    lea rdx, [files]
    mov [rdx+rax*8], rsi
    inc qword [nfiles]
.nextarg:
    inc rbx
    jmp .arg
.args:
    call cfg_args
    lea rcx, [k_out]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov [outdir], rax
    cmp qword [nfiles], 0
    je .usage
    lea rcx, [k_src]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    lea rcx, [c_audio]
    mov [colname+C_AUDIO*8], rcx
    cmp dword [rax], 'vox'
    je .vox
    cmp word [rax], 'ls'
    jne .usage
    lea rcx, [l_text]
    mov [colname+C_TEXT*8], rcx
    lea rcx, [v_spk]
    mov [colname+C_SPK*8], rcx
    lea rcx, [l_id]
    mov [colname+C_ID*8], rcx
    jmp .go
.vox:
    mov qword [isvox], 1
    lea rcx, [v_text]
    mov [colname+C_TEXT*8], rcx
    lea rcx, [v_norm]
    mov [colname+C_NORM*8], rcx
    lea rcx, [v_spk]
    mov [colname+C_SPK*8], rcx
    lea rcx, [v_id]
    mov [colname+C_ID*8], rcx
    lea rcx, [v_gold]
    mov [colname+C_GOLD*8], rcx
.go:
    mov rcx, [outdir]
    call make_path
    lea rcx, [k_thr]
    mov edx, 4
    call cfg_int
    mov ecx, eax
    call pool_init
    call time_now
    mov [rsp+32], rax
    lea rcx, [task]
    xor edx, edx
    mov r8, [nfiles]
    mov r9d, 1
    call par_for
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+32], xmm0
    mov rcx, [nfiles]
    call print_dec
    say " files -> shards: "
    mov rcx, [utts]
    call print_dec
    say " utterances, "
    cvtsi2sd xmm0, qword [samples]
    divsd xmm0, [c_rate]
    divsd xmm0, [c_hour]
    mov edx, 2
    call print_fixed
    say " h in "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " s", 13, 10, "  not gold: "
    mov rcx, [notgold]
    call print_dec
    say ", speaker id vs the id's prefix mismatches: "
    mov rcx, [spkbad]
    call print_dec
    say ", unusable audio: "
    mov rcx, [failed]
    call print_dec
    say ", rows skipped (nulls): "
    mov rcx, [skipped]
    call print_dec
    say 13, 10
    call con_restore
    mov rcx, [failed]
    call ExitProcess
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
