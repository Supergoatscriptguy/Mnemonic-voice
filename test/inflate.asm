; stage 3 test: crc32, inflate on hand-made streams, gzip on the real manifests
; uses: data\inflate
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "data/inflate.inc"

section .rdata
s9      db "123456789"
; fixed huffman, final: literal 'a', end of block
d_a     db 0x4b, 0x04, 0x00
; fixed: 'a', then length 9 at distance 1, end. worked out bit by bit
d_a10   db 0x4b, 0x84, 0x03, 0x00
; stored, final: "hello"
d_st    db 0x01, 0x05, 0x00, 0xfa, 0xff, "hello"
; stored (not final) "hello", then the fixed 'a' block
d_two   db 0x00, 0x05, 0x00, 0xfa, 0xff, "hello", 0x4b, 0x04, 0x00
; stored with a bad ~LEN
d_nlen  db 0x01, 0x05, 0x00, 0xfa, 0xfe, "hello"
; fixed: a match before anything was written
d_back  db 0x83, 0x03, 0x00
; the fixed 'a' block cut off after its first byte
d_cut   db 0x4b
; gzip of "a"
g_a     db 0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 0x4b, 0x04, 0x00
        db 0x43, 0xbe, 0xb7, 0xe8, 1, 0, 0, 0
g_bad   db 0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 0x4b, 0x04, 0x00
        db 0x44, 0xbe, 0xb7, 0xe8, 1, 0, 0, 0
x_a10   db "aaaaaaaaaa"
x_two   db "helloa"
f_small db "datasets\libriheavy\cuts_small.jsonl.gz", 0
f_dev   db "datasets\libriheavy\cuts_dev.jsonl.gz", 0
f_tc    db "datasets\libriheavy\cuts_test_clean.jsonl.gz", 0
f_to    db "datasets\libriheavy\cuts_test_other.jsonl.gz", 0
align 8
c_mb    dq 1e-6

section .bss
alignb 8
inf     resb IF_SIZE
lines   resq 1
bytes   resq 1

section .text

; rcx = stream, edx = length. inflates it whole into a 64 KB window.
; eax = what inflate_run said at the end
raw:
    push rbx
    sub rsp, 32
    mov r8d, edx
    mov rdx, rcx
    lea rcx, [inf]
    mov r9d, 65536
    call inflate_init
.run:
    lea rcx, [inf]
    call inflate_run
    cmp eax, 1
    je .run
    add rsp, 32
    pop rbx
    ret

; [inf]'s window vs rcx/edx. eax = 1 if it's exactly that
same:
    push rsi
    push rdi
    xor eax, eax
    cmp [inf+IF_N], rdx
    jne .out
    mov rsi, [inf+IF_OUT]
    mov rdi, rcx
    mov ecx, edx
    repe cmpsb
    sete al
.out:
    pop rdi
    pop rsi
    ret

; rcx = path, edx = window. gunzips the whole file, counting lines.
; eax = gz_check, and bytes/lines are filled in
gz_file:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov r12d, edx
    call file_map
    test rax, rax
    jz .bad
    mov rbx, rax
    lea rcx, [inf]
    mov r8, rdx
    mov rdx, rax
    mov r9d, r12d
    call gz_open
    test eax, eax
    jz .bad
    xor esi, esi
.run:
    lea rcx, [inf]
    call inflate_run
    mov edi, eax
    cmp edi, -1
    je .bad
    mov rcx, [inf+IF_FROM]
    mov rdx, [inf+IF_N]
    mov r8, [inf+IF_OUT]
.nl:
    cmp rcx, rdx
    jae .counted
    cmp byte [r8+rcx], 10
    jne .nn
    inc rsi
.nn:
    inc rcx
    jmp .nl
.counted:
    cmp edi, 1
    je .run
    mov [lines], rsi
    mov rax, [inf+IF_TOTAL]
    add rax, [inf+IF_N]
    mov [bytes], rax
    lea rcx, [inf]
    call gz_check
    mov edi, eax
    lea rcx, [inf]
    call inflate_free
    mov rcx, rbx
    call file_unmap
    mov eax, edi
    jmp .out
.bad:
    xor eax, eax
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; file, window. gunzips it, checks the trailer, prints size, lines and speed
%macro gzcheck 2
    call time_now
    mov [rsp+32], rax
    lea rcx, [%1]
    mov edx, %2
    call gz_file
    mov ebx, eax
    mov rcx, [rsp+32]
    call time_since
    movsd [rsp+40], xmm0
    say "  "
    lea rcx, [%1]
    call print_z
    say ", window "
    mov ecx, %2
    call print_dec
    say ": "
    mov rcx, [bytes]
    call print_dec
    say " bytes, "
    mov rcx, [lines]
    call print_dec
    say " lines, "
    cvtsi2sd xmm0, qword [bytes]
    divsd xmm0, [rsp+40]
    mulsd xmm0, [c_mb]
    xor edx, edx
    call print_fixed
    say " MB/s", 13, 10
    cmp ebx, 1
    check e, "    crc32 and length match"
%endmacro

global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init

    say "crc32", 13, 10
    xor ecx, ecx
    lea rdx, [s9]
    mov r8d, 9
    call crc32
    cmp eax, 0xcbf43926
    check e, "  crc32(123456789) = cbf43926"
    xor ecx, ecx
    lea rdx, [s9]
    xor r8d, r8d
    call crc32
    cmp eax, 0
    check e, "  crc32 of nothing = 0"
    xor ecx, ecx
    lea rdx, [s9]
    mov r8d, 4
    call crc32
    mov ecx, eax
    lea rdx, [s9+4]
    mov r8d, 5
    call crc32
    cmp eax, 0xcbf43926
    check e, "  in two pieces, the same"

    say "inflate, hand-made streams", 13, 10
    lea rcx, [d_a]
    mov edx, 3
    call raw
    test eax, eax
    jnz .a_bad
    cmp qword [inf+IF_N], 1
    jne .a_bad
    mov rax, [inf+IF_OUT]
    cmp byte [rax], 'a'
.a_bad:
    check e, "  fixed huffman: 'a'"
    lea rcx, [d_a10]
    mov edx, 4
    call raw
    lea rcx, [x_a10]
    mov edx, 10
    call same
    cmp eax, 1
    check e, "  'a' + length 9 at distance 1 = 10 a's (overlapping copy)"
    lea rcx, [d_st]
    mov edx, 10
    call raw
    lea rcx, [x_two]
    mov edx, 5
    call same
    cmp eax, 1
    check e, "  stored block: hello"
    lea rcx, [d_two]
    mov edx, 13
    call raw
    lea rcx, [x_two]
    mov edx, 6
    call same
    cmp eax, 1
    check e, "  stored then fixed: helloa"
    lea rcx, [d_nlen]
    mov edx, 10
    call raw
    cmp eax, -1
    check e, "  stored with a bad ~LEN is refused"
    lea rcx, [d_back]
    mov edx, 3
    call raw
    cmp eax, -1
    check e, "  a match reaching before the start is refused"
    lea rcx, [d_cut]
    mov edx, 1
    call raw
    cmp eax, -1
    check e, "  a stream cut off mid block is refused"

    say "gzip", 13, 10
    lea rcx, [inf]
    lea rdx, [g_a]
    mov r8d, 21
    mov r9d, 65536
    call gz_open
.ga:
    lea rcx, [inf]
    call inflate_run
    cmp eax, 1
    je .ga
    lea rcx, [inf]
    call gz_check
    cmp eax, 1
    check e, "  gzip of 'a': crc and length check out"
    lea rcx, [inf]
    lea rdx, [g_bad]
    mov r8d, 21
    mov r9d, 65536
    call gz_open
.gb:
    lea rcx, [inf]
    call inflate_run
    cmp eax, 1
    je .gb
    lea rcx, [inf]
    call gz_check
    cmp eax, 0
    check e, "  a wrong crc is caught"

    say "the libriheavy manifests (crc32 + length from each gzip trailer)", 13, 10
    gzcheck f_small, 4 << 20
    cmp qword [lines], 122526
    check e, "    122526 cuts"
    cmp qword [bytes], 423213364
    check e, "    423213364 bytes"
    gzcheck f_small, 65536
    gzcheck f_dev, 4 << 20
    gzcheck f_tc, 4 << 20
    gzcheck f_to, 4 << 20
    jmp t_done
