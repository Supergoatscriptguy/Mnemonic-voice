; sptok train [out=datasets\sptok.bin] DIR...   the speech bpe, learned from the cleaned
;                                               transcripts in DIR\*.shard (shards whose
;                                               names start with dev, test or valid are skipped)
; sptok show [tok=datasets\sptok.bin] TEXT...   the text cleaned and cut into its tokens
; sptok stats [tok=datasets\sptok.bin] DIR...   what the cleanup leaves out, tokens per second
;                                               of audio, and utterances too dense for ctc
;                                               (80 ms frames: tokens + repeats > frames)
; uses: tokenizer\pretok tokenizer\bpe tokenizer\tok asr\text data\shard
default rel
bits 64
%include "lib.inc"
%include "tokenizer/tok.inc"
%include "data/shard.inc"

extern ExitProcess, clean_text

section .rdata
k_out   db "out", 0
k_tok   db "tok", 0
d_tok   db "datasets\sptok.bin", 0
s_pat   db "\*.shard", 0
s_vocab db ".vocab.txt", 0
s_tmp   db ".tmp", 0
m_train db "train", 0
m_show  db "show", 0
m_stats db "stats", 0
usage   db "usage: sptok train [out=...] DIR... | show [tok=...] TEXT... | stats [tok=...] DIR...", 13, 10, 0
e_tok   db "can't load the tokenizer", 0
e_write db "couldn't write the tokenizer", 0
align 8
c_rate  dq 16000.0

section .bss
alignb 8
job     resb BT_SIZE
hdr     resb TKH_SIZE
sr      resb SR_SIZE
texta   resb ARENA_SIZE
docsa   resb ARENA_SIZE
docs    resq 1
ndocs   resq 1
nutt    resq 1
dropped resq 1
tbytes  resq 1
ntok    resq 1
nsamp   resq 1
dense   resq 1
worst   resq 1                  ; most tokens per second (x 1000)
ctx     resq 1
outp    resq 1
pattern resb 1024
found   resb 1 << 20
tmpp    resb 1024
vpath   resb 1024
buf     resb 1 << 16
toks    resw 1 << 16
line    resb 4096

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

; rcx = directory, rdx = callback(rbx = SR, r13 = utterance index), r8d = 1 for
; training shards only. runs the callback on every utterance
walk:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov r14, rdx
    mov [rsp+32], r8
    mov rdx, rcx
    lea rcx, [pattern]
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
    cmp qword [rsp+32], 0
    je .open
    mov rcx, [rsi]
    call training
    test eax, eax
    jz .next
.open:
    mov rcx, [rsi]
    lea rdx, [sr]
    call shard_open
    test eax, eax
    jz .next
    lea rbx, [sr]
    xor r13d, r13d
.utt:
    cmp r13, [sr+SR_COUNT]
    jae .next
    call r14
    inc r13
    jmp .utt
.next:
    add rsi, 8
    dec rdi
    jmp .shard
.out:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rbx = SR, r13 = utterance. rax = its cleaned text in buf, rdx = length (-1 = left out)
cleaned:
    sub rsp, 40
    imul rax, r13, UT_SIZE
    add rax, [rbx+SR_INDEX]
    mov rcx, [rax+UT_TEXT]
    add rcx, [rbx+SR_BASE]
    mov edx, [rax+UT_TLEN]
    cmp edx, (1 << 16) / 3
    jb .fits
    mov rdx, -1
    jmp .out
.fits:
    lea r8, [buf]
    call clean_text
    mov rdx, rax
    lea rax, [buf]
.out:
    add rsp, 40
    ret

; walk callback for train: keeps the cleaned text as a doc
keep:
    push rsi
    push rdi
    sub rsp, 56
    ; locals: length +32, where the copy went +40
    inc qword [nutt]
    call cleaned
    cmp rdx, -1
    jne .kept
    inc qword [dropped]
    jmp .out
.kept:
    mov [rsp+32], rdx
    add [tbytes], rdx
    lea rcx, [texta]
    mov r8d, 1
    call arena_alloc
    mov [rsp+40], rax
    mov rdi, rax
    lea rsi, [buf]
    mov rcx, [rsp+32]
    rep movsb
    lea rcx, [docsa]
    mov edx, 16
    mov r8d, 8
    call arena_alloc
    mov rcx, [rsp+40]
    mov [rax], rcx
    mov rcx, [rsp+32]
    mov [rax+8], rcx
    cmp qword [ndocs], 0
    jne .more
    mov [docs], rax
.more:
    inc qword [ndocs]
.out:
    add rsp, 56
    pop rdi
    pop rsi
    ret

; walk callback for stats: tokens per second, and whether ctc can fit them
measure:
    push rsi
    push rdi
    push r12
    sub rsp, 32
    inc qword [nutt]
    call cleaned
    cmp rdx, -1
    jne .tok
    inc qword [dropped]
    jmp .out
.tok:
    mov rcx, [ctx]
    mov r8, rdx
    mov rdx, rax
    lea r9, [toks]
    call tok_encode
    mov rsi, rax                ; tokens
    add [ntok], rax
    imul rax, r13, UT_SIZE
    add rax, [rbx+SR_INDEX]
    mov r12d, [rax+UT_N]
    add [nsamp], r12
    ; ctc needs a frame per token, plus one between equal neighbours
    mov rdi, rsi
    xor ecx, ecx
    lea rdx, [toks]
.rep:
    lea eax, [rcx+1]
    cmp rax, rsi
    jae .counted
    mov ax, [rdx+rcx*2]
    cmp ax, [rdx+rcx*2+2]
    jne .nr
    inc rdi
.nr:
    inc ecx
    jmp .rep
.counted:
    mov eax, r12d
    add eax, 1279
    xor edx, edx
    mov ecx, 1280
    div ecx                     ; 80 ms frames
    cmp rdi, rax
    jbe .fits
    inc qword [dense]
.fits:
    ; tokens per second, x 1000
    mov rax, rsi
    imul rax, rax, 16000000
    xor edx, edx
    test r12, r12
    jz .out
    div r12
    cmp rax, [worst]
    jbe .out
    mov [worst], rax
.out:
    add rsp, 32
    pop r12
    pop rdi
    pop rsi
    ret

; rcx = path. writes the vocab, one token a line, next to the tokenizer
vocab:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rdx, rcx
    lea rcx, [vpath]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_vocab]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [vpath]
    call file_create
    mov r12, rax
    xor ebx, ebx
.t:
    cmp ebx, SPECIAL0
    jae .done
    lea rdi, [line]
    mov rcx, rdi
    mov edx, ebx
    call fmt_dec
    mov rdi, rax
    mov byte [rdi], 9
    mov byte [rdi+1], '['
    add rdi, 2
    mov ecx, ebx
    call tok_bytes
    mov rsi, rax
    mov rcx, rdx
.b:
    test rcx, rcx
    jz .eol
    lodsb
    cmp al, 32
    jb .hex
    cmp al, 127
    je .hex
    stosb
    dec rcx
    jmp .b
.hex:
    mov byte [rdi], 92
    mov byte [rdi+1], 'x'
    movzx edx, al
    shr edx, 4
    add edx, '0'
    cmp edx, '9'
    jbe .h1
    add edx, 'a' - '9' - 1
.h1:
    mov [rdi+2], dl
    movzx edx, al
    and edx, 15
    add edx, '0'
    cmp edx, '9'
    jbe .h2
    add edx, 'a' - '9' - 1
.h2:
    mov [rdi+3], dl
    add rdi, 4
    dec rcx
    jmp .b
.eol:
    mov byte [rdi], ']'
    mov word [rdi+1], 0x0a0d
    add rdi, 3
    mov rcx, r12
    lea rdx, [line]
    mov r8, rdi
    sub r8, rdx
    call file_write
    inc ebx
    jmp .t
.done:
    mov rcx, r12
    call file_close
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
    push rdi
    push r12
    sub rsp, 40
    call lib_init
    cmp qword [argc], 3
    jb .usage
    ; the plain args (dirs or texts) are noted before cfg_args cuts the others up
    lea rax, [argv]
    mov rsi, [rax+8]            ; the mode
    call cfg_args
    mov rcx, rsi
    lea rdx, [m_train]
    call same
    test eax, eax
    jnz .train
    mov rcx, rsi
    lea rdx, [m_show]
    call same
    test eax, eax
    jnz .show
    mov rcx, rsi
    lea rdx, [m_stats]
    call same
    test eax, eax
    jnz .stats
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess

.train:
    lea rcx, [k_out]
    lea rdx, [d_tok]
    call cfg_str
    mov [outp], rax
    lea rcx, [texta]
    mov rdx, 1 << 32
    call arena_init
    lea rcx, [texta]            ; whash/memeq read up to 8 bytes before a chunk, so
    mov edx, 64                 ; the first text can't sit right at the arena's start
    mov r8d, 1
    call arena_alloc
    lea rcx, [docsa]
    mov rdx, 1 << 30
    call arena_init
    mov r12d, 2
.tdir:
    cmp r12, [argc]
    jae .tdone
    lea rax, [argv]
    mov rcx, [rax+r12*8]
    call plain
    test eax, eax
    jz .tnext
    lea rax, [argv]
    mov rcx, [rax+r12*8]
    lea rdx, [keep]
    mov r8d, 1
    call walk
.tnext:
    inc r12
    jmp .tdir
.tdone:
    mov rcx, [nutt]
    call print_dec
    say " transcripts, "
    mov rcx, [dropped]
    call print_dec
    say " left out by the cleanup, "
    mov rcx, [tbytes]
    call print_dec
    say " bytes of text to learn from", 13, 10
    xor ecx, ecx
    call pool_init
    mov rax, [docs]
    mov [job+BT_DOCS], rax
    mov rax, [ndocs]
    mov [job+BT_NDOCS], rax
    mov qword [job+BT_NMERGES], NMERGES
    mov qword [job+BT_VERBOSE], 1
    mov ecx, NMERGES * 4 + 8    ; whash reads a little before the start
    call mem_alloc
    add rax, 8
    mov [job+BT_MERGES], rax
    mov ecx, NMERGES * 8
    call mem_alloc
    mov [job+BT_COUNTS], rax
    call time_now
    mov rbx, rax
    lea rcx, [job]
    call bpe_train
    say "  "
    mov rcx, [job+BT_DONE]
    call print_dec
    say " merges in "
    mov rcx, rbx
    call time_since
    mov edx, 2
    call print_fixed
    say " s, consistency problems: "
    mov rcx, [job+BT_BAD]
    call print_dec
    say 13, 10
    mov rax, TK_MAGIC
    mov [hdr+TKH_MAGIC], rax
    mov dword [hdr+TKH_VOCAB], VOCAB
    mov rax, [job+BT_DONE]
    mov [hdr+TKH_NMERGES], eax
    mov dword [hdr+TKH_SPECIAL0], SPECIAL0
    mov dword [hdr+TKH_NSPECIAL], NSPECIAL
    mov rcx, [job+BT_MERGES]
    mov rdx, [job+BT_DONE]
    shl rdx, 2
    call whash
    mov [hdr+TKH_HASH], rax
    lea rcx, [tmpp]
    mov rdx, [outp]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_tmp]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [tmpp]
    call file_create
    cmp rax, -1
    je .werr
    mov rbx, rax
    mov rcx, rbx
    lea rdx, [hdr]
    mov r8d, TKH_SIZE
    call file_write
    mov rcx, rbx
    mov rdx, [job+BT_MERGES]
    mov r8, [job+BT_DONE]
    shl r8, 2
    call file_write
    test eax, eax
    jz .werr
    mov rcx, rbx
    call file_flush
    mov rcx, rbx
    call file_close
    lea rcx, [tmpp]
    mov rdx, [outp]
    call file_replace
    test eax, eax
    jz .werr
    mov rcx, [outp]
    call tok_load
    test eax, eax
    jz .terr
    mov rcx, [outp]
    call vocab
    say "  wrote "
    mov rcx, [outp]
    call print_z
    say " and "
    lea rcx, [vpath]
    call print_z
    say 13, 10
    jmp .bye

.show:
    call loadtok
    mov r12d, 2
.sarg:
    cmp r12, [argc]
    jae .bye
    lea rax, [argv]
    mov rsi, [rax+r12*8]
    mov rcx, rsi
    call plain
    test eax, eax
    jz .snext
    mov rcx, rsi
.slen:
    cmp byte [rcx], 0
    je .slend
    inc rcx
    jmp .slen
.slend:
    mov rdx, rcx
    sub rdx, rsi
    mov rcx, rsi
    lea r8, [buf]
    call clean_text
    cmp rax, -1
    jne .sclean
    say "  (left out)", 13, 10
    jmp .snext
.sclean:
    mov rdi, rax
    say "  "
    lea rcx, [buf]
    mov rdx, rdi
    call print
    say 13, 10, "  "
    mov rcx, [ctx]
    lea rdx, [buf]
    mov r8, rdi
    lea r9, [toks]
    call tok_encode
    mov rbx, rax
    xor edi, edi
.piece:
    cmp rdi, rbx
    jae .pdone
    lea rax, [toks]
    movzx ecx, word [rax+rdi*2]
    call tok_bytes
    mov rcx, rax
    call print
    say "|"
    inc rdi
    jmp .piece
.pdone:
    say "  ("
    mov rcx, rbx
    call print_dec
    say " tokens)", 13, 10
.snext:
    inc r12
    jmp .sarg

.stats:
    call loadtok
    mov r12d, 2
.dir:
    cmp r12, [argc]
    jae .sdone
    lea rax, [argv]
    mov rcx, [rax+r12*8]
    call plain
    test eax, eax
    jz .dnext
    lea rax, [argv]
    mov rcx, [rax+r12*8]
    lea rdx, [measure]
    xor r8d, r8d
    call walk
.dnext:
    inc r12
    jmp .dir
.sdone:
    mov rcx, [nutt]
    call print_dec
    say " utterances, "
    mov rcx, [dropped]
    call print_dec
    say " left out by the cleanup, "
    mov rcx, [ntok]
    call print_dec
    say " tokens, "
    cvtsi2sd xmm0, qword [ntok]
    cvtsi2sd xmm1, qword [nsamp]
    divsd xmm1, [c_rate]
    divsd xmm0, xmm1
    mov edx, 2
    call print_fixed
    say " a second on average, the densest "
    cvtsi2sd xmm0, qword [worst]
    mov eax, 1000
    cvtsi2sd xmm1, eax
    divsd xmm0, xmm1
    mov edx, 1
    call print_fixed
    say ". too dense for ctc at 80 ms: "
    mov rcx, [dense]
    call print_dec
    say 13, 10
.bye:
    call con_restore
    xor ecx, ecx
    call ExitProcess
.werr:
    lea rcx, [e_write]
    call fatal
.terr:
    lea rcx, [e_tok]
    call fatal

; the tokenizer named by tok= (or the default), and an encode context
loadtok:
    sub rsp, 40
    lea rcx, [k_tok]
    lea rdx, [d_tok]
    call cfg_str
    mov rcx, rax
    call tok_load
    test eax, eax
    jnz .ok
    lea rcx, [e_tok]
    call fatal
.ok:
    call tok_cache_new
    mov [ctx], rax
    add rsp, 40
    ret

; rcx, rdx = two zero terminated strings. eax = 1 if they're the same
same:
    mov al, [rcx]
    cmp al, [rdx]
    jne .no
    test al, al
    jz .yes
    inc rcx
    inc rdx
    jmp same
.yes:
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret
