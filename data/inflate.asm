; inflate (rfc 1951) and the gzip wrapper (rfc 1952). streams through an output
; window: each inflate_run decodes until the window is nearly full and hands back
; the new bytes, keeping the last 32 KB as history for matches
default rel
bits 64
%include "lib.inc"
%include "data/inflate.inc"

WIN     equ 32768
SLACK   equ 64                  ; matches copy 8 bytes at a time and may overshoot

section .rdata
align 4
; crc32 (0xedb88320, reflected), the table built at assembly time
crctab:
%assign i 0
%rep 256
    %assign c i
    %rep 8
        %if c & 1
            %assign c (c >> 1) ^ 0xedb88320
        %else
            %assign c c >> 1
        %endif
    %endrep
    dd c
    %assign i i+1
%endrep

; length symbols 257..285: base and extra bits
lbase   dw 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31
        dw 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
lext    db 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2
        db 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
align 2
dbase   dw 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193
        dw 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
dext    db 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
        db 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
; the order code length code lengths come in
clord   db 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15

section .text

; rcx = crc so far (0 to start), rdx = bytes, r8 = count. eax = crc
global crc32
crc32:
    mov eax, ecx
    not eax
    lea r9, [crctab]
    test r8, r8
    jz .out
.b:
    movzx ecx, byte [rdx]
    xor ecx, eax
    and ecx, 255
    shr eax, 8
    xor eax, [r9+rcx*4]
    inc rdx
    dec r8
    jnz .b
.out:
    not eax
    ret

; the slow way to the next 8 bytes, for the stream's last few: rax = byte index,
; rsi = input, r13 = its length. rax = the bytes that are there, zeros past the end
peek_slow:
    push rcx
    push rdx
    mov rdx, rax
    xor eax, eax
    mov ecx, 0
.b:
    cmp rdx, r13
    jae .done
    movzx r8d, byte [rsi+rdx]
    shl r8, cl
    or rax, r8
    inc rdx
    add ecx, 8
    cmp ecx, 64
    jb .b
.done:
    pop rdx
    pop rcx
    ret

; rax = 64 bits from bit position r12 (rsi = input, r13 = its length). trashes rcx, rdx
%macro peek 0
    mov rax, r12
    shr rax, 3
    lea rdx, [rax+8]
    cmp rdx, r13
    ja %%slow
    mov rax, [rsi+rax]
    jmp %%got
%%slow:
    call peek_slow
%%got:
    mov ecx, r12d
    and ecx, 7
    shr rax, cl
%endmacro

; builds a decode table from code lengths. rcx = lengths (bytes), edx = how many,
; r8 = table (u16 per index). index = the next bits of the stream, so codes go in
; bit reversed. eax = index bits (the longest code), -1 if the lengths overflow
build:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 136                ; counts[16] u32 at +0, next[16] u32 at +64
    mov rsi, rcx
    mov r12d, edx
    mov r13, r8
    xor eax, eax
    mov ecx, 32
    lea rdi, [rsp]
    rep stosd
    xor ecx, ecx
    xor ebx, ebx                ; longest
.count:
    cmp ecx, r12d
    jae .counted
    movzx eax, byte [rsi+rcx]
    inc dword [rsp+rax*4]
    cmp eax, ebx
    cmova ebx, eax
    inc ecx
    jmp .count
.counted:
    mov dword [rsp], 0
    ; canonical codes: next[len] = (next[len-1] + count[len-1]) << 1
    xor eax, eax
    mov ecx, 1
.next:
    add eax, [rsp+rcx*4-4]
    add eax, eax
    mov [rsp+64+rcx*4], eax
    inc ecx
    cmp ecx, 16
    jb .next
    ; codes of a length mustn't run past 2^len
    mov ecx, 1
.over:
    mov eax, [rsp+64+rcx*4]
    add eax, [rsp+rcx*4]
    mov edx, 1
    shl edx, cl
    cmp eax, edx
    ja .bad
    inc ecx
    cmp ecx, 16
    jb .over
    ; clear the table, 2^longest entries (an incomplete code leaves holes = 0)
    mov ecx, ebx
    mov eax, 1
    shl eax, cl
    mov ecx, eax
    mov rdi, r13
    xor eax, eax
    rep stosw
    xor r14d, r14d              ; symbol
.sym:
    cmp r14d, r12d
    jae .built
    movzx ecx, byte [rsi+r14]
    test ecx, ecx
    jz .nextsym
    mov eax, [rsp+64+rcx*4]
    inc dword [rsp+64+rcx*4]
    ; reverse len bits of the code
    xor edx, edx
    mov r8d, ecx
.rev:
    add edx, edx
    mov r9d, eax
    and r9d, 1
    or edx, r9d
    shr eax, 1
    dec r8d
    jnz .rev
    ; every index whose low len bits are the code
    mov r9d, r14d
    shl r9d, 4
    or r9d, ecx
    mov r8d, 1
    shl r8d, cl                 ; step
    mov r10d, 1
    mov r11d, ecx
    mov ecx, ebx
    shl r10d, cl                ; table size
    mov ecx, r11d
.fill:
    cmp edx, r10d
    jae .nextsym
    mov [r13+rdx*2], r9w
    add edx, r8d
    jmp .fill
.nextsym:
    inc r14d
    jmp .sym
.built:
    mov eax, ebx
    jmp .out
.bad:
    mov eax, -1
.out:
    add rsp, 136
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = IF, rdx = deflate stream, r8 = its length, r9 = window size (>= 64 KB)
global inflate_init
inflate_init:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    mov [rbx+IF_IN], rdx
    mov [rbx+IF_INLEN], r8
    mov [rbx+IF_CAP], r9
    xor eax, eax
    mov [rbx+IF_POS], rax
    mov [rbx+IF_N], rax
    mov [rbx+IF_FROM], rax
    mov [rbx+IF_TOTAL], rax
    mov [rbx+IF_STATE], eax
    mov [rbx+IF_LAST], eax
    mov [rbx+IF_CRC], eax
    lea rcx, [r9+SLACK]
    call mem_alloc
    mov [rbx+IF_OUT], rax
    mov ecx, 65536 * 2
    call mem_alloc
    mov [rbx+IF_LTAB], rax
    add rax, 65536
    mov [rbx+IF_DTAB], rax
    add rsp, 32
    pop rbx
    ret

; rcx = IF
global inflate_free
inflate_free:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    mov rcx, [rbx+IF_OUT]
    call mem_free
    mov rcx, [rbx+IF_LTAB]
    call mem_free
    add rsp, 32
    pop rbx
    ret

; fixed huffman tables: lit/len 0-143 8 bits, 144-255 9, 256-279 7, 280-287 8;
; 32 distance codes of 5 bits. rbx = IF
fixed:
    push rdi
    sub rsp, 32
    lea rdi, [rbx+IF_LENS]
    mov al, 8
    mov ecx, 144
    rep stosb
    mov al, 9
    mov ecx, 112
    rep stosb
    mov al, 7
    mov ecx, 24
    rep stosb
    mov al, 8
    mov ecx, 8
    rep stosb
    lea rcx, [rbx+IF_LENS]
    mov edx, 288
    mov r8, [rbx+IF_LTAB]
    call build
    mov [rbx+IF_LBITS], eax
    lea rdi, [rbx+IF_LENS]
    mov al, 5
    mov ecx, 32
    rep stosb
    lea rcx, [rbx+IF_LENS]
    mov edx, 32
    mov r8, [rbx+IF_DTAB]
    call build
    mov [rbx+IF_DBITS], eax
    add rsp, 32
    pop rdi
    ret

; reads a dynamic block's code lengths and builds its tables. rbx = IF, rsi = input,
; r12 = bit position, r13 = input length. eax = 1, or 0 if the header is broken
dynamic:
    push rdi
    push r14
    push r15
    sub rsp, 64                 ; 19 code length code lengths at +32
    peek
    mov r14d, eax
    and r14d, 31
    add r14d, 257               ; HLIT
    mov r15d, eax
    shr r15d, 5
    and r15d, 31
    inc r15d                    ; HDIST
    mov edi, eax
    shr edi, 10
    and edi, 15
    add edi, 4                  ; HCLEN
    add r12, 14
    cmp r14d, 286
    ja .bad
    cmp r15d, 30
    ja .bad
    xor eax, eax
    mov [rsp+32], rax
    mov [rsp+40], rax
    mov [rsp+48], rax
    peek                        ; 4..19 x 3 bits = at most 57
    lea r8, [clord]
    xor ecx, ecx
.cl:
    movzx edx, byte [r8+rcx]
    mov r9d, eax
    and r9d, 7
    mov [rsp+32+rdx], r9b
    shr rax, 3
    inc ecx
    cmp ecx, edi
    jb .cl
    lea rax, [rdi+rdi*2]
    add r12, rax
    lea rcx, [rsp+32]
    mov edx, 19
    mov r8, [rbx+IF_DTAB]       ; borrowed for the code length code
    call build
    cmp eax, -1
    je .bad
    mov edi, eax                ; its index bits
    ; the HLIT + HDIST code lengths, run length coded
    lea r9, [rbx+IF_LENS]
    xor r10d, r10d              ; filled
    lea r11d, [r14+r15]         ; wanted
.len:
    cmp r10d, r11d
    jae .lens
    peek
    mov ecx, edi
    bzhi edx, eax, ecx
    mov r8, [rbx+IF_DTAB]
    movzx edx, word [r8+rdx*2]
    mov ecx, edx
    and ecx, 15
    jz .bad
    shr edx, 4
    add r12, rcx
    shr rax, cl
    cmp edx, 16
    jb .plain
    je .repeat
    cmp edx, 17
    je .zeros3
    ; 18: 11-138 zeros
    mov ecx, eax
    and ecx, 127
    add ecx, 11
    add r12, 7
    jmp .zero
.zeros3:
    mov ecx, eax
    and ecx, 7
    add ecx, 3
    add r12, 3
.zero:
    lea eax, [r10+rcx]
    cmp eax, r11d
    ja .bad
.z:
    mov byte [r9+r10], 0
    inc r10d
    dec ecx
    jnz .z
    jmp .len
.repeat:
    test r10d, r10d
    jz .bad
    movzx edx, byte [r9+r10-1]
    mov ecx, eax
    and ecx, 3
    add ecx, 3
    add r12, 2
    lea eax, [r10+rcx]
    cmp eax, r11d
    ja .bad
.r:
    mov [r9+r10], dl
    inc r10d
    dec ecx
    jnz .r
    jmp .len
.plain:
    mov [r9+r10], dl
    inc r10d
    jmp .len
.lens:
    cmp byte [r9+256], 0        ; there has to be an end of block code
    je .bad
    mov rcx, r9
    mov edx, r14d
    mov r8, [rbx+IF_LTAB]
    call build
    cmp eax, -1
    je .bad
    mov [rbx+IF_LBITS], eax
    lea rcx, [rbx+IF_LENS]
    add rcx, r14
    mov edx, r15d
    mov r8, [rbx+IF_DTAB]
    call build
    cmp eax, -1
    je .bad
    mov [rbx+IF_DBITS], eax
    mov eax, 1
    jmp .out
.bad:
    xor eax, eax
.out:
    add rsp, 64
    pop r15
    pop r14
    pop rdi
    ret

; rcx = IF. decodes until the window is nearly full or the stream ends.
; new output is [IF_FROM, IF_N). eax = 1 more to come, 0 that was the end,
; -1 the stream is broken
global inflate_run
inflate_run:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov rbx, rcx
    ; slide: keep the last 32 KB as history
    mov rcx, [rbx+IF_N]
    cmp rcx, WIN
    jbe .slid
    mov rdi, [rbx+IF_OUT]
    lea rsi, [rdi+rcx-WIN]
    sub rcx, WIN
    add [rbx+IF_TOTAL], rcx
    mov ecx, WIN / 8
    rep movsq
    mov qword [rbx+IF_N], WIN
.slid:
    mov rax, [rbx+IF_N]
    mov [rbx+IF_FROM], rax
    mov rsi, [rbx+IF_IN]
    mov r12, [rbx+IF_POS]
    mov r13, [rbx+IF_INLEN]
    mov rdi, [rbx+IF_OUT]
    mov r11, [rbx+IF_N]
    mov eax, [rbx+IF_STATE]
    cmp eax, 1
    je .stored
    cmp eax, 2
    je .huff
    cmp eax, 3
    je .end

.block:
    cmp dword [rbx+IF_LAST], 0
    jne .finished
    mov rax, r12
    shr rax, 3
    cmp rax, r13
    jae .bad                    ; ran out of input before the final block
    peek
    mov ecx, eax
    and ecx, 1
    mov [rbx+IF_LAST], ecx
    shr eax, 1
    and eax, 3
    add r12, 3
    cmp eax, 1
    je .fix
    cmp eax, 2
    je .dyn
    test eax, eax
    jnz .bad
    ; stored: byte aligned LEN, ~LEN, then the bytes
    add r12, 7
    and r12, -8
    mov rax, r12
    shr rax, 3
    lea rdx, [rax+4]
    cmp rdx, r13
    ja .bad
    movzx ecx, word [rsi+rax]
    movzx edx, word [rsi+rax+2]
    not edx
    and edx, 0xffff
    cmp ecx, edx
    jne .bad
    mov [rbx+IF_LEFT], rcx
    add r12, 32
    mov dword [rbx+IF_STATE], 1
.stored:
    mov rcx, [rbx+IF_LEFT]
    test rcx, rcx
    jz .storeddone
    mov rdx, [rbx+IF_CAP]
    sub rdx, r11
    jz .full
    cmp rcx, rdx
    cmova rcx, rdx
    mov rax, r12
    shr rax, 3
    lea rdx, [rax+rcx]
    cmp rdx, r13
    ja .bad
    sub [rbx+IF_LEFT], rcx
    lea rdx, [rcx*8]
    add r12, rdx
    push rsi
    push rdi
    lea rsi, [rsi+rax]
    lea rdi, [rdi+r11]
    add r11, rcx
    rep movsb
    pop rdi
    pop rsi
    jmp .stored
.storeddone:
    mov dword [rbx+IF_STATE], 0
    jmp .block

.fix:
    mov [rbx+IF_N], r11         ; build trashes r8-r11
    call fixed
    mov r11, [rbx+IF_N]
    jmp .codes
.dyn:
    mov [rbx+IF_N], r11
    call dynamic
    mov r11, [rbx+IF_N]
    test eax, eax
    jz .bad
.codes:
    mov dword [rbx+IF_STATE], 2
.huff:
    mov r14, [rbx+IF_LTAB]
    mov r15, [rbx+IF_DTAB]
.sym:
    lea rax, [r11+258]
    cmp rax, [rbx+IF_CAP]
    ja .full
    peek
    mov ecx, [rbx+IF_LBITS]
    bzhi edx, eax, ecx
    movzx edx, word [r14+rdx*2]
    mov ecx, edx
    and ecx, 15
    jz .bad
    shr edx, 4
    add r12, rcx
    shr rax, cl
    cmp edx, 256
    jb .lit
    je .eob
    sub edx, 257
    cmp edx, 28
    ja .bad
    lea r8, [lbase]
    movzx r8d, word [r8+rdx*2]
    lea r9, [lext]
    movzx ecx, byte [r9+rdx]
    bzhi r9d, eax, ecx
    add r8d, r9d                ; length
    add r12, rcx
    shr rax, cl
    mov ecx, [rbx+IF_DBITS]
    bzhi edx, eax, ecx
    movzx edx, word [r15+rdx*2]
    mov ecx, edx
    and ecx, 15
    jz .bad
    shr edx, 4
    add r12, rcx
    shr rax, cl
    cmp edx, 29
    ja .bad
    lea r9, [dbase]
    movzx r9d, word [r9+rdx*2]
    lea r10, [dext]
    movzx ecx, byte [r10+rdx]
    bzhi r10d, eax, ecx
    add r9d, r10d               ; distance
    add r12, rcx
    mov rax, [rbx+IF_TOTAL]
    add rax, r11
    cmp r9, rax
    ja .bad                     ; reaches back before the start
    ; copy, overlapping on purpose when distance < length
    lea rdx, [rdi+r11]
    mov rcx, rdx
    sub rcx, r9
    add r11, r8
    cmp r9d, 8
    jb .bytes
.q:
    mov rax, [rcx]
    mov [rdx], rax
    add rcx, 8
    add rdx, 8
    sub r8d, 8
    jg .q
    jmp .sym
.bytes:
    mov al, [rcx]
    mov [rdx], al
    inc rcx
    inc rdx
    dec r8d
    jnz .bytes
    jmp .sym
.lit:
    mov [rdi+r11], dl
    inc r11
    jmp .sym
.eob:
    mov dword [rbx+IF_STATE], 0
    jmp .block

.finished:
    mov dword [rbx+IF_STATE], 3
.end:
    xor r14d, r14d
    jmp .past
.full:
    mov r14d, 1
.past:
    ; past the end the reader just sees zeros, so a cut off stream shows up here
    mov rax, r13
    shl rax, 3
    cmp r12, rax
    ja .bad
.save:
    mov [rbx+IF_POS], r12
    mov [rbx+IF_N], r11
    mov ecx, [rbx+IF_CRC]
    mov rdx, [rbx+IF_FROM]
    mov r8, r11
    sub r8, rdx
    add rdx, rdi
    call crc32
    mov [rbx+IF_CRC], eax
    mov eax, r14d
    jmp .out
.bad:
    mov [rbx+IF_POS], r12
    mov [rbx+IF_N], r11
    mov eax, -1
.out:
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = IF, rdx = gzip file bytes, r8 = size, r9 = window size. reads the gzip
; header and starts inflating what follows. eax = 1, 0 if it isn't gzip
global gz_open
gz_open:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rcx
    mov rsi, rdx
    cmp r8, 18
    jb .no
    cmp word [rsi], 0x8b1f
    jne .no
    cmp byte [rsi+2], 8         ; deflate
    jne .no
    movzx eax, byte [rsi+3]     ; flags
    lea rdx, [rsi+10]
    lea r10, [rsi+r8]
    test eax, 4                 ; FEXTRA
    jz .name
    movzx ecx, word [rdx]
    lea rdx, [rdx+rcx+2]
.name:
    test eax, 8                 ; FNAME
    jz .comment
.n:
    cmp rdx, r10
    jae .no
    inc rdx
    cmp byte [rdx-1], 0
    jne .n
.comment:
    test eax, 16                ; FCOMMENT
    jz .hcrc
.c:
    cmp rdx, r10
    jae .no
    inc rdx
    cmp byte [rdx-1], 0
    jne .c
.hcrc:
    test eax, 2                 ; FHCRC
    jz .body
    add rdx, 2
.body:
    mov r8, r10
    sub r8, rdx
    jb .no
    mov rcx, rbx
    call inflate_init
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = IF, after inflate_run said 0. eax = 1 if the gzip trailer's crc32 and
; length match what came out
global gz_check
gz_check:
    mov rax, [rcx+IF_POS]
    add rax, 7
    shr rax, 3
    lea rdx, [rax+8]
    cmp rdx, [rcx+IF_INLEN]
    ja .no
    mov rdx, [rcx+IF_IN]
    mov r8d, [rdx+rax]
    cmp r8d, [rcx+IF_CRC]
    jne .no
    mov r8d, [rdx+rax+4]
    mov r9, [rcx+IF_TOTAL]
    add r9, [rcx+IF_N]
    cmp r8d, r9d                ; the length mod 2^32
    jne .no
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret
