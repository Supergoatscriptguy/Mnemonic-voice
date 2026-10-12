; lmbuild DIR... [out=datasets\lm.bin] [min2=2] [min3=2]
; the word trigram model from the training shards' transcripts (never dev*, test*
; or valid*): every word, pair and triple counted, each utterance padded with two
; starts and an end; pairs and triples seen fewer than min2 / min3 times left out,
; the rest written as the tables lm_load reads
; uses: data\shard asr\lm
default rel
bits 64
%include "lib.inc"
%include "data/shard.inc"
%include "asr/lm.inc"

extern ExitProcess

; counting tables: 2^L slots each
L1      equ 23
L2      equ 26
L3      equ 27

section .rdata
k_out   db "out", 0
k_min2  db "min2", 0
k_min3  db "min3", 0
d_out   db "datasets\lm.bin", 0
s_pat   db "\*.shard", 0
usage   db "usage: lmbuild DIR... [out=datasets\lm.bin] [min2=2] [min3=2]", 13, 10, 0
e_full  db "a count table filled up", 0
e_write db "can't write the model", 0
names   db "words", 0, "pairs", 0, "triples", 0

section .bss
alignb 8
dirs    resq 64
ndirs   resq 1
pattern resb 600
sr      resb SR_SIZE
tk      resq 3                  ; counting tables: keys, counts, slots - 1, used
tc      resq 3
tm      resq 3
tn      resq 3
mins    resq 3
nkept   resq 3
ok      resq 3                  ; the kept tables
oc      resq 3
on      resq 3
words   resq 1
utts    resq 1
nshards resq 1
w0      resq 1                  ; the two words before the next
w1      resq 1
hdr     resb LH_SIZE

section .text

; ecx = table, rax = key: one more of it. touches rax, rcx, rdx, r8-r11
bump:
    lea r8, [tk]
    mov r9, [r8+rcx*8]          ; keys
    lea r8, [tm]
    mov rdx, [r8+rcx*8]         ; mask
    mov r10, rax
.p:
    and r10, rdx
    mov r11, [r9+r10*8]
    test r11, r11
    jz .new
    cmp r11, rax
    je .inc
    inc r10
    jmp .p
.new:
    mov [r9+r10*8], rax
    lea r8, [tn]
    inc qword [r8+rcx*8]
    mov r11, [r8+rcx*8]
    shl r11, 2
    lea r8, [tm]
    mov r8, [r8+rcx*8]
    imul r8, r8, 3              ; fill to three quarters at most
    cmp r11, r8
    jb .inc
    sub rsp, 8
    lea rcx, [e_full]
    call fatal
.inc:
    lea r8, [tc]
    mov r8, [r8+rcx*8]
    inc dword [r8+r10*4]
    ret

; rax = the next word: it, its pair with w1 and its triple with w0 w1 counted
next_word:
    push rbx
    sub rsp, 32
    mov rbx, rax
    xor ecx, ecx
    call bump
    mov rcx, [w1]
    mov rdx, rbx
    call lm_key2
    mov ecx, 1
    call bump
    mov rcx, [w0]
    mov rdx, [w1]
    mov r8, rbx
    call lm_key3
    mov ecx, 2
    call bump
    mov rax, [w1]
    mov [w0], rax
    mov [w1], rbx
    inc qword [words]
    add rsp, 32
    pop rbx
    ret

; rcx = text, rdx = its length: one utterance counted
utterance:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    mov rdi, rdx
    ; the start, as a word, and the (start, start) pair, once each
    mov rax, LM_START
    mov [w0], rax
    mov [w1], rax
    xor ecx, ecx
    call bump
    mov rcx, LM_START
    mov rdx, rcx
    call lm_key2
    mov ecx, 1
    call bump
    inc qword [words]
    xor ebx, ebx                ; 1 while in a word
    call lm_wstart
    mov r8, rax
.b:
    test rdi, rdi
    jz .end
    movzx ecx, byte [rsi]
    lea rax, [lm_wtab]
    movzx ecx, byte [rax+rcx]
    inc rsi
    dec rdi
    test ecx, ecx
    jz .sep
    xor r8, rcx
    mov rax, 0x100000001b3
    imul r8, rax
    mov ebx, 1
    jmp .b
.sep:
    test ebx, ebx
    jz .b
    mov rax, r8
    call next_word
    xor ebx, ebx
    call lm_wstart
    mov r8, rax
    jmp .b
.end:
    test ebx, ebx
    jz .fin
    mov rax, r8
    call next_word
.fin:
    mov rax, LM_END
    call next_word
    inc qword [utts]
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = directory: every training shard's transcripts
directory:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    lea rcx, [pattern]
    mov rdx, rsi
    call fmt_str
    mov rcx, rax
    lea rdx, [s_pat]
    call fmt_str
    mov byte [rax], 0
    mov ecx, 1 << 20
    call mem_alloc
    mov rbx, rax
    lea rcx, [pattern]
    mov rdx, rbx
    mov r8d, 1 << 20
    call file_find
    mov rdi, rax
    mov rsi, rdx
.shard:
    test rdi, rdi
    jz .out
    mov rcx, [rsi]
    call training
    test eax, eax
    jz .next
    mov rcx, [rsi]
    lea rdx, [sr]
    call shard_open
    test eax, eax
    jz .next
    inc qword [nshards]
    push rsi
    push rdi
    sub rsp, 32
    xor esi, esi
.u:
    cmp rsi, [sr+SR_COUNT]
    jae .closed
    imul rax, rsi, UT_SIZE
    add rax, [sr+SR_INDEX]
    mov rcx, [rax+UT_TEXT]
    add rcx, [sr+SR_BASE]
    mov edx, [rax+UT_TLEN]
    call utterance
    inc rsi
    jmp .u
.closed:
    add rsp, 32
    pop rdi
    pop rsi
    lea rcx, [sr]
    call shard_close
    say 13, "  "
    mov rcx, [nshards]
    call print_dec
    say " shards, "
    mov rcx, [utts]
    call print_dec
    say " utterances, "
    mov rcx, [words]
    call print_dec
    say " words", 27, "[K"
.next:
    add rsi, 8
    dec rdi
    jmp .shard
.out:
    mov rcx, rbx
    call mem_free
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = a shard's path. eax = 1 unless it's held out (dev*, test*, valid*)
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

; ecx = table: its entries seen at least mins[table] times into a new table twice
; their number (a power of two)
keep:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov ebx, ecx
    lea rax, [tk]
    mov rsi, [rax+rbx*8]
    lea rax, [tc]
    mov rdi, [rax+rbx*8]
    lea rax, [mins]
    mov r12d, [rax+rbx*8]
    lea rax, [tm]
    mov r13, [rax+rbx*8]
    ; how many stay
    xor eax, eax
    xor ecx, ecx
.c:
    cmp rcx, r13
    ja .counted
    cmp dword [rdi+rcx*4], r12d
    jb .cn
    inc rax
.cn:
    inc rcx
    jmp .c
.counted:
    lea rcx, [nkept]
    mov [rcx+rbx*8], rax
    add rax, rax
    mov ecx, 1024
    cmp rax, rcx
    cmovb rax, rcx
    dec rax
    bsr rcx, rax
    inc ecx
    mov eax, 1
    shl rax, cl
    lea rcx, [on]
    mov [rcx+rbx*8], rax
    mov [rsp+32], rax           ; the slots
    lea rcx, [rax*8]
    call mem_alloc
    lea rcx, [ok]
    mov [rcx+rbx*8], rax
    mov rcx, [rsp+32]
    shl rcx, 2
    call mem_alloc
    lea rcx, [oc]
    mov [rcx+rbx*8], rax
    lea rax, [ok]
    mov r8, [rax+rbx*8]
    lea rax, [oc]
    mov r9, [rax+rbx*8]
    mov r10, [rsp+32]
    dec r10                     ; new mask
    xor ecx, ecx
.i:
    cmp rcx, r13
    ja .done
    mov eax, [rdi+rcx*4]
    cmp eax, r12d
    jb .in
    mov rdx, [rsi+rcx*8]
    mov r11, rdx
.p:
    and r11, r10
    cmp qword [r8+r11*8], 0
    je .put
    inc r11
    jmp .p
.put:
    mov [r8+r11*8], rdx
    mov [r9+r11*4], eax
.in:
    inc rcx
    jmp .i
.done:
    ; the counting table can go
    mov rcx, rsi
    call mem_free
    mov rcx, rdi
    call mem_free
    add rsp, 48
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
    push rdi
    sub rsp, 48
    call lib_init
    ; directories are the args without '='
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
    jz .dir
    cmp al, '='
    je .nextarg
    inc rcx
    jmp .eq
.dir:
    mov rax, [ndirs]
    lea rdx, [dirs]
    mov [rdx+rax*8], rsi
    inc qword [ndirs]
.nextarg:
    inc rbx
    jmp .arg
.args:
    call cfg_args
    cmp qword [ndirs], 0
    jne .go
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
.go:
    mov qword [mins], 1
    lea rcx, [k_min2]
    mov edx, 2
    call cfg_int
    mov [mins+8], rax
    lea rcx, [k_min3]
    mov edx, 2
    call cfg_int
    mov [mins+16], rax
    ; the counting tables
    xor ebx, ebx
.t:
    lea rax, [tbits]
    movzx ecx, byte [rax+rbx]
    mov eax, 1
    shl rax, cl
    lea rcx, [tm]
    lea rdx, [rax-1]
    mov [rcx+rbx*8], rdx
    mov rsi, rax
    lea rcx, [rax*8]
    call mem_alloc
    lea rcx, [tk]
    mov [rcx+rbx*8], rax
    lea rcx, [rsi*4]
    call mem_alloc
    lea rcx, [tc]
    mov [rcx+rbx*8], rax
    inc ebx
    cmp ebx, 3
    jb .t
    ; count
    call time_now
    mov rdi, rax
    xor ebx, ebx
.d:
    cmp rbx, [ndirs]
    jae .counted
    lea rax, [dirs]
    mov rcx, [rax+rbx*8]
    call directory
    inc rbx
    jmp .d
.counted:
    say 13, 10, "  counted in "
    mov rcx, rdi
    call time_since
    mov edx, 1
    call print_fixed
    say " s. distinct "
    mov rcx, [tn]
    call print_dec
    say " words, "
    mov rcx, [tn+8]
    call print_dec
    say " pairs, "
    mov rcx, [tn+16]
    call print_dec
    say " triples", 13, 10
    ; prune, then write: header, then keys and counts of each table
    xor ebx, ebx
.k:
    mov ecx, ebx
    call keep
    inc ebx
    cmp ebx, 3
    jb .k
    say "  kept "
    mov rcx, [nkept]
    call print_dec
    say " words, "
    mov rcx, [nkept+8]
    call print_dec
    say " pairs (seen "
    mov rcx, [mins+8]
    call print_dec
    say "+ times), "
    mov rcx, [nkept+16]
    call print_dec
    say " triples ("
    mov rcx, [mins+16]
    call print_dec
    say "+)", 13, 10
    mov rax, LM_MAGIC
    mov [hdr+LH_MAGIC], rax
    mov rax, [words]
    mov [hdr+LH_WORDS], rax
    mov rax, [on]
    mov [hdr+LH_N1], rax
    mov rax, [on+8]
    mov [hdr+LH_N2], rax
    mov rax, [on+16]
    mov [hdr+LH_N3], rax
    mov eax, LH_SIZE
    xor ebx, ebx
    lea rsi, [hdr+LH_K1]
.off:
    mov [rsi], rax
    lea rcx, [on]
    mov rdx, [rcx+rbx*8]
    lea rax, [rax+rdx*8]
    mov [rsi+8], rax
    lea rax, [rax+rdx*4]
    add rsi, 16
    inc ebx
    cmp ebx, 3
    jb .off
    mov [rsp+40], rax           ; the file's size
    lea rcx, [k_out]
    lea rdx, [d_out]
    call cfg_str
    mov rsi, rax
    mov rcx, rax
    call file_create
    cmp rax, -1
    jne .made
    lea rcx, [e_write]
    call fatal
.made:
    mov rdi, rax
    mov rcx, rdi
    lea rdx, [hdr]
    mov r8d, LH_SIZE
    call file_write
    xor ebx, ebx
.w:
    mov rcx, rdi
    lea rax, [ok]
    mov rdx, [rax+rbx*8]
    lea rax, [on]
    mov r8, [rax+rbx*8]
    shl r8, 3
    call file_write
    test eax, eax
    jz .bad
    mov rcx, rdi
    lea rax, [oc]
    mov rdx, [rax+rbx*8]
    lea rax, [on]
    mov r8, [rax+rbx*8]
    shl r8, 2
    call file_write
    test eax, eax
    jz .bad
    inc ebx
    cmp ebx, 3
    jb .w
    mov rcx, rdi
    call file_close
    say "  wrote "
    mov rcx, rsi
    call print_z
    say ", "
    mov rax, [rsp+40]
    shr rax, 20
    mov rcx, rax
    call print_dec
    say " MB", 13, 10
    call con_restore
    xor ecx, ecx
    call ExitProcess
.bad:
    lea rcx, [e_write]
    call fatal

section .rdata
tbits   db L1, L2, L3
