; writing shards (format in shard.inc). sw_open, then sw_add per utterance (it
; encodes the flac), then sw_close writes the index and trailer and renames the
; .tmp into place. one SW per thread
default rel
bits 64
%include "lib.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

WBUF    equ 8 << 20

section .rdata
s_tmp   db ".tmp", 0
e_write db "a shard write failed", 0

section .text

; rcx = escaped json string, rdx = its end, r8 = out. writes it as utf-8,
; rax = the end of what got written (never longer than the input)
global json_unesc
json_unesc:
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

; 4 hex digits at rcx. eax = value
hex4:
    xor eax, eax
    mov r9d, 4
.d:
    movzx edx, byte [rcx]
    inc rcx
    or edx, 0x20                ; lower case, digits don't care
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

; rbx = SW. writes out the buffer
flushb:
    sub rsp, 40
    mov rcx, [rbx+SW_FILE]
    mov rdx, [rbx+SW_BUF]
    mov r8, [rbx+SW_N]
    test r8, r8
    jz .out
    call file_write
    test eax, eax
    jnz .ok
    lea rcx, [e_write]
    call fatal
.ok:
    mov rax, [rbx+SW_N]
    add [rbx+SW_OFF], rax
    mov qword [rbx+SW_N], 0
.out:
    add rsp, 40
    ret

; rbx = SW, rcx = bytes about to go in. flushes first if they don't fit.
; rax = where they go, rdx = their file offset
room:
    sub rsp, 40
    mov rax, [rbx+SW_N]
    add rax, rcx
    cmp rax, WBUF
    jbe .fits
    call flushb
.fits:
    mov rax, [rbx+SW_BUF]
    add rax, [rbx+SW_N]
    mov rdx, [rbx+SW_OFF]
    add rdx, [rbx+SW_N]
    add rsp, 40
    ret

; rbx = SW, rcx = bytes, rdx = how many. rax = their file offset. anything too big
; for the buffer (minutes of audio) goes straight to the file
copyin:
    push rsi
    push rdi
    sub rsp, 40
    mov rsi, rcx
    mov rdi, rdx
    cmp rdx, WBUF / 2
    jb .buffered
    call flushb
    mov rax, [rbx+SW_OFF]
    mov [rsp+32], rax
    mov rcx, [rbx+SW_FILE]
    mov rdx, rsi
    mov r8, rdi
    call file_write
    test eax, eax
    jnz .direct
    lea rcx, [e_write]
    call fatal
.direct:
    add [rbx+SW_OFF], rdi
    mov rax, [rsp+32]
    add rsp, 40
    pop rdi
    pop rsi
    ret
.buffered:
    mov rcx, rdx
    call room
    mov [rsp+32], rdx
    mov rcx, rdi
    mov rdi, rax
    add [rbx+SW_N], rcx
    rep movsb
    mov rax, [rsp+32]
    add rsp, 40
    pop rdi
    pop rsi
    ret

; rcx = SW (zeroed), rdx = the shard's path. eax = 1 if the .tmp got created
global sw_open
sw_open:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    lea rcx, [rbx+SW_PATH]
    call fmt_str
    mov byte [rax], 0
    lea rcx, [rbx+SW_TMP]
    lea rdx, [rbx+SW_PATH]
    call fmt_str
    mov rcx, rax
    lea rdx, [s_tmp]
    call fmt_str
    mov byte [rax], 0
    mov ecx, WBUF
    call mem_alloc
    mov [rbx+SW_BUF], rax
    mov qword [rbx+SW_CAP], 4096
    mov ecx, 4096 * UT_SIZE
    call mem_alloc
    mov [rbx+SW_IDX], rax
    lea rcx, [rbx+SW_TMP]
    call file_create
    mov [rbx+SW_FILE], rax
    xor eax, eax
    cmp qword [rbx+SW_FILE], -1
    setne al
    add rsp, 32
    pop rbx
    ret

; rcx = SW, rdx = UA (the utterance). encodes the samples as flac and writes it
; with its texts and id
global sw_add
sw_add:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    ; a bigger index if it's full
    mov rax, [rbx+SW_NIDX]
    cmp rax, [rbx+SW_CAP]
    jb .roomy
    imul rcx, rax, UT_SIZE * 2
    call mem_alloc
    mov r12, rax
    mov rdi, rax
    push rsi
    mov rsi, [rbx+SW_IDX]
    imul rcx, [rbx+SW_NIDX], UT_SIZE
    rep movsb
    pop rsi
    mov rcx, [rbx+SW_IDX]
    mov [rbx+SW_IDX], r12
    call mem_free
    shl qword [rbx+SW_CAP], 1
.roomy:
    mov rdi, [rbx+SW_NIDX]
    imul rdi, rdi, UT_SIZE
    add rdi, [rbx+SW_IDX]       ; the entry
    ; the audio
    mov rcx, [rsi+UA_PCM]
    mov rdx, [rsi+UA_N]
    mov r8d, 16000
    lea r9, [rbx+SW_FE]
    call flac_encode
    mov rcx, [rbx+SW_FE+FE_DATA]
    mov rdx, [rbx+SW_FE+FE_LEN]
    call copyin
    mov [rdi+UT_FLAC], rax
    mov rax, [rbx+SW_FE+FE_LEN]
    mov [rdi+UT_FLEN], eax
    mov rcx, [rbx+SW_FE+FE_DATA]
    call mem_free
    mov rax, [rsi+UA_N]
    mov [rdi+UT_N], eax
    add [rbx+SW_SAMP], rax
    ; the texts: copied as they are, or json unescaped
    lea r12, [rsi+UA_TEXT]
    mov ecx, UT_TEXT
    call .text
    lea r12, [rsi+UA_NORM]
    mov ecx, UT_NORM
    call .text
    mov rcx, [rsi+UA_ID]
    mov edx, [rsi+UA_ILEN]
    call copyin
    mov [rdi+UT_ID], rax
    mov eax, [rsi+UA_ILEN]
    mov [rdi+UT_ILEN], eax
    mov eax, [rsi+UA_SPK]
    mov [rdi+UT_SPK], eax
    mov eax, [rsi+UA_SRC]
    mov [rdi+UT_SRC], eax
    inc qword [rbx+SW_NIDX]
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
; r12 = the UA's (ptr, len) pair, ecx = the UT field it goes in (the length after it)
.text:
    push r13
    sub rsp, 32
    mov r13d, ecx
    cmp dword [rsi+UA_ESC], 0
    jne .esc
    mov rcx, [r12]
    mov edx, [r12+8]
    call copyin
    mov [rdi+r13], rax
    mov eax, [r12+8]
    mov [rdi+r13+8], eax
    jmp .tdone
.esc:
    mov ecx, [r12+8]
    call room
    mov [rdi+r13], rdx
    mov r8, rax
    mov rcx, [r12]
    mov edx, [r12+8]
    add rdx, rcx
    push r8
    sub rsp, 8
    call json_unesc
    add rsp, 8
    pop r8
    sub rax, r8
    mov [rdi+r13+8], eax
    add [rbx+SW_N], rax
.tdone:
    add rsp, 32
    pop r13
    ret

; rcx = SW. index, trailer, flush, rename into place, free. eax = 1
global sw_close
sw_close:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    mov rbx, rcx
    call flushb
    mov rax, [rbx+SW_OFF]
    mov [rsp+32], rax           ; where the index starts
    mov r8, [rbx+SW_NIDX]
    imul r8, r8, UT_SIZE
    mov rsi, r8
    test r8, r8
    jz .noidx
    mov rcx, [rbx+SW_FILE]
    mov rdx, [rbx+SW_IDX]
    call file_write
    test eax, eax
    jz .bad
    add [rbx+SW_OFF], rsi
.noidx:
    mov rdi, [rbx+SW_BUF]
    xor eax, eax
    mov ecx, SH_SIZE / 8
    rep stosq
    mov rdi, [rbx+SW_BUF]
    mov rax, 'MVSHARD1'
    mov [rdi+SH_MAGIC], rax
    mov rax, [rbx+SW_NIDX]
    mov [rdi+SH_COUNT], rax
    mov rax, [rsp+32]
    mov [rdi+SH_INDEX], rax
    mov rax, [rbx+SW_SAMP]
    mov [rdi+SH_SAMPLES], rax
    mov dword [rdi+SH_RATE], 16000
    mov qword [rbx+SW_N], SH_SIZE
    call flushb
    mov rcx, [rbx+SW_FILE]
    call file_flush
    mov rcx, [rbx+SW_FILE]
    call file_close
    lea rcx, [rbx+SW_TMP]
    lea rdx, [rbx+SW_PATH]
    call file_replace
    test eax, eax
    jz .bad
    mov rcx, [rbx+SW_IDX]
    call mem_free
    mov rcx, [rbx+SW_BUF]
    call mem_free
    mov eax, 1
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret
.bad:
    lea rcx, [e_write]
    call fatal
