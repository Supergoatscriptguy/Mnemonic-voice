; flac encoder for our shards: 16 bit mono, 4096 sample blocks. each block gets
; the best fixed predictor (orders 0-4, by the sum of its residuals) with rice
; partitions picked per block, or constant / verbatim when those are smaller. the
; output is a whole valid .flac (STREAMINFO with the md5, crc-8 and crc-16 on every
; frame) so anything can play it. no lpc yet: it would save maybe 5-10%
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern crc8tab, crc16tab

BLOCK   equ 4096
MAXP    equ 6                   ; the finest rice partition order tried
RES     equ BLOCK * 4           ; bytes of one order's residuals in the scratch
SUMS    equ 5 * RES             ; then the partition sums (u64, up to 64 of them)
SCRATCH equ SUMS + 64 * 8

section .text

; ---- the bit writer: rdi = next byte, r14 = pending bits (right aligned),
; r15d = how many. msb first

; ecx = n (0-32), rdx = value (its low n bits). trashes rax, rcx, rdx
global flac_put
flac_put:
put:
    test ecx, ecx
    jz .done
    shl r14, cl
    bzhi rdx, rdx, rcx
    or r14, rdx
    add r15d, ecx
.flush:
    cmp r15d, 8
    jb .done
    sub r15d, 8
    mov rax, r14
    mov ecx, r15d
    shr rax, cl
    mov [rdi], al
    inc rdi
    jmp .flush
.done:
    ret

%macro putc 2
    mov ecx, %1
    mov edx, %2
    call put
%endmacro

; edx = u, r9d = k. one rice code: u >> k zeros, a 1, then u's low k bits.
; trashes rax, rcx, rdx, r8
rice:
    mov eax, edx
    mov ecx, r9d
    shr eax, cl                 ; q
    lea r8d, [rax+r9+1]
    cmp r8d, 32
    ja .long
    bzhi edx, edx, ecx
    bts edx, ecx                ; the 1 that ends the unary part
    mov ecx, r8d
    jmp put
.long:
    push rdx
    mov r8d, eax
.zeros:
    cmp r8d, 32
    jb .rest
    xor edx, edx
    mov ecx, 32
    call put
    sub r8d, 32
    jmp .zeros
.rest:
    lea ecx, [r8+1]
    mov edx, 1
    call put
    pop rdx
    mov ecx, r9d
    bzhi edx, edx, ecx
    jmp put

; r8 = samples, r9 = the sum of their zigzag values. eax = the rice parameter with
; the fewest bits, by the usual estimate m (k + 1) + (S >> k), rdx = those bits
bestk:
    push rbx
    xor ebx, ebx
    mov r10, -1
    xor r11d, r11d
.k:
    mov ecx, ebx
    mov rax, r9
    shr rax, cl
    lea rdx, [rbx+1]
    imul rdx, r8
    add rax, rdx
    cmp rax, r10
    jae .worse
    mov r10, rax
    mov r11d, ebx
.worse:
    inc ebx
    cmp ebx, 30
    jbe .k
    mov eax, r11d
    mov rdx, r10
    pop rbx
    ret

; edx = predictor order, ecx = partition order, ebp = block size, r12 = scratch.
; the zigzagged residuals of each partition, summed into the scratch's sums
psums:
    push rsi
    mov r9d, ebp
    shr r9d, cl                 ; partition size
    mov rsi, rdx
    shl rsi, 14                 ; RES = 1 << 14
    add rsi, r12
    lea r11, [r12+SUMS]
    mov ecx, edx                ; the first partition starts after the warmup
    mov edx, r9d                ; where the current one ends
.part:
    xor eax, eax
.add:
    cmp ecx, edx
    jae .end
    mov r8d, [rsi+rcx*4]
    lea r10d, [r8+r8]
    sar r8d, 31
    xor r10d, r8d
    add rax, r10
    inc ecx
    jmp .add
.end:
    mov [r11], rax
    add r11, 8
    add edx, r9d
    cmp ecx, ebp
    jb .part
    pop rsi
    ret

; ecx = partition order, edx = predictor order, ebp = block size, r12 = scratch.
; rax = estimated bits of the residual coded that way (headers included),
; edx = 1 if it needs rice2 (a parameter over 14)
pcost:
    push rbx
    push rsi
    push rdi
    push r13
    sub rsp, 56
    ; locals: bits +32, highest k +40
    mov r13d, ecx
    mov ebx, edx
    call psums
    mov esi, 1
    mov ecx, r13d
    shl esi, cl                 ; partitions
    mov edi, ebp
    shr edi, cl                 ; their size
    xor eax, eax
    mov [rsp+32], rax
    mov [rsp+40], rax
    xor r13d, r13d              ; partition
.p:
    mov r8d, edi
    test r13d, r13d
    jnz .m
    sub r8d, ebx                ; the first is short by the warmup
.m:
    mov r9, [r12+SUMS+r13*8]
    call bestk
    add [rsp+32], rdx
    cmp eax, [rsp+40]
    jbe .lower
    mov [rsp+40], eax
.lower:
    inc r13d
    cmp r13d, esi
    jb .p
    xor edx, edx
    mov eax, 4
    cmp dword [rsp+40], 14
    jbe .pbits
    mov edx, 1
    mov eax, 5
.pbits:
    imul eax, esi               ; every partition carries its parameter
    add rax, [rsp+32]
    add rax, 6                  ; method and partition order
    add rsp, 56
    pop r13
    pop rdi
    pop rsi
    pop rbx
    ret

; one block's subframe. rsi = int16 samples, ebp = how many, r12 = scratch, and
; the bit writer (rdi, r14, r15)
subframe_enc:
    push rbx
    push r13
    sub rsp, 88
    ; locals: S0..S4 +32, order +72, partition order +76, rice2 +80
    ; all the same: one constant
    movsx ebx, word [rsi]
    mov ecx, 1
.cst:
    cmp ecx, ebp
    jae .constant
    movsx edx, word [rsi+rcx*2]
    cmp edx, ebx
    jne .predict
    inc ecx
    jmp .cst
.constant:
    putc 8, 0
    mov ecx, 16
    mov edx, ebx
    call put
    jmp .out

.predict:
    ; residuals of the fixed predictors = successive differences
    xor ecx, ecx
.d0:
    movsx eax, word [rsi+rcx*2]
    mov [r12+rcx*4], eax
    inc ecx
    cmp ecx, ebp
    jb .d0
    mov r8d, 1
.dk:
    mov r9, r8
    shl r9, 14
    add r9, r12                 ; this order
    lea r10, [r9-RES]           ; the one below
    mov ecx, r8d
.dki:
    cmp ecx, ebp
    jae .dknext
    mov eax, [r10+rcx*4]
    sub eax, [r10+rcx*4-4]
    mov [r9+rcx*4], eax
    inc ecx
    jmp .dki
.dknext:
    inc r8d
    cmp r8d, 5
    jb .dk
    ; the order with the smallest residual (that the block is long enough for)
    xor r8d, r8d
.sk:
    mov r9, r8
    shl r9, 14
    add r9, r12
    xor r11d, r11d
    mov ecx, r8d
.ski:
    cmp ecx, ebp
    jae .skdone
    mov eax, [r9+rcx*4]
    lea edx, [rax+rax]
    sar eax, 31
    xor edx, eax
    add r11, rdx
    inc ecx
    jmp .ski
.skdone:
    mov [rsp+32+r8*8], r11
    inc r8d
    cmp r8d, 5
    jb .sk
    xor ebx, ebx
    mov r11, [rsp+32]
    mov r8d, 1
.bo:
    cmp r8d, 5
    jae .chosen
    cmp r8d, ebp
    jae .chosen
    mov rax, [rsp+32+r8*8]
    cmp rax, r11
    jae .bonext
    mov r11, rax
    mov ebx, r8d
.bonext:
    inc r8d
    jmp .bo
.chosen:
    mov [rsp+72], ebx
    ; partition orders: the finest that splits the block evenly with room for
    ; the warmup, then every coarser one, keeping the cheapest
    mov r13d, MAXP
.fine:
    mov ecx, r13d
    mov eax, ebp
    shr eax, cl
    shl eax, cl
    cmp eax, ebp
    jne .coarser
    mov eax, ebp
    shr eax, cl
    cmp eax, ebx
    ja .search
.coarser:
    dec r13d
    jmp .fine
.search:
    mov qword [rsp+56], -1      ; reuses S3/S4's slots: they're done with
.try:
    mov ecx, r13d
    mov edx, ebx
    call pcost
    cmp rax, [rsp+56]
    jae .notbetter
    mov [rsp+56], rax
    mov [rsp+76], r13d
    mov [rsp+80], edx
.notbetter:
    dec r13d
    jns .try
    ; verbatim instead, if that's smaller
    mov rax, [rsp+56]
    mov ecx, ebx
    shl ecx, 4
    add rax, rcx                ; the warmup
    mov ecx, ebp
    shl ecx, 4
    cmp rax, rcx
    jb .fixed
    putc 8, 2
    xor ebx, ebx
.verb:
    movzx edx, word [rsi+rbx*2]
    mov ecx, 16
    call put
    inc ebx
    cmp ebx, ebp
    jb .verb
    jmp .out

.fixed:
    lea edx, [rbx+8]
    add edx, edx
    mov ecx, 8
    call put                    ; type: fixed, this order, no wasted bits
    xor r13d, r13d
.warm:
    cmp r13d, ebx
    jae .res
    movzx edx, word [rsi+r13*2]
    mov ecx, 16
    call put
    inc r13d
    jmp .warm
.res:
    mov ecx, 2
    mov edx, [rsp+80]
    call put                    ; 0 rice, 1 rice2
    mov ecx, 4
    mov edx, [rsp+76]
    call put
    mov ecx, [rsp+76]
    mov edx, ebx
    call psums
    ; each partition: its parameter, then its codes
    mov r13, rbx
    shl r13, 14
    add r13, r12
    mov ecx, [rsp+76]
    mov eax, ebp
    shr eax, cl
    mov [rsp+32], eax           ; partition size
    mov dword [rsp+36], 0       ; partition
    mov [rsp+40], ebx           ; next sample
.part:
    mov ecx, [rsp+76]
    mov eax, 1
    shl eax, cl
    cmp [rsp+36], eax
    jae .out
    mov r8d, [rsp+32]
    cmp dword [rsp+36], 0
    jne .full
    sub r8d, ebx
.full:
    mov [rsp+44], r8d           ; codes in this partition
    mov eax, [rsp+36]
    mov r9, [r12+SUMS+rax*8]
    call bestk
    mov [rsp+48], eax
    mov ecx, 4
    add ecx, [rsp+80]
    mov edx, eax
    call put
    mov r9d, [rsp+48]
.code:
    cmp dword [rsp+44], 0
    je .nextpart
    mov eax, [rsp+40]
    mov eax, [r13+rax*4]
    lea edx, [rax+rax]
    sar eax, 31
    xor edx, eax
    call rice
    inc dword [rsp+40]
    dec dword [rsp+44]
    jmp .code
.nextpart:
    inc dword [rsp+36]
    jmp .part
.out:
    add rsp, 88
    pop r13
    pop rbx
    ret

; edx = n < 2^21, the frame number, coded the utf-8 way
utf8:
    push rbx
    mov ebx, edx
    cmp ebx, 0x80
    jae .two
    mov ecx, 8
    call put
    pop rbx
    ret
.two:
    cmp ebx, 0x800
    jae .three
    mov edx, ebx
    shr edx, 6
    or edx, 0xc0
    mov ecx, 8
    call put
    jmp .last
.three:
    cmp ebx, 0x10000
    jae .four
    mov edx, ebx
    shr edx, 12
    or edx, 0xe0
    mov ecx, 8
    call put
    jmp .mid
.four:
    mov edx, ebx
    shr edx, 18
    or edx, 0xf0
    mov ecx, 8
    call put
    mov edx, ebx
    shr edx, 12
    and edx, 63
    or edx, 0x80
    mov ecx, 8
    call put
.mid:
    mov edx, ebx
    shr edx, 6
    and edx, 63
    or edx, 0x80
    mov ecx, 8
    call put
.last:
    mov edx, ebx
    and edx, 63
    or edx, 0x80
    mov ecx, 8
    call put
    pop rbx
    ret

; rcx = from, rdx = to. eax = crc-8 of those bytes
global flac_crc8
flac_crc8:
crc8:
    xor eax, eax
    lea r8, [crc8tab]
.b:
    cmp rcx, rdx
    jae .out
    xor al, [rcx]
    movzx eax, byte [r8+rax]
    inc rcx
    jmp .b
.out:
    ret

; rcx = from, rdx = to. eax = crc-16 of those bytes
global flac_crc16
flac_crc16:
crc16:
    xor eax, eax
    lea r8, [crc16tab]
.b:
    cmp rcx, rdx
    jae .out
    mov r9d, eax
    shr r9d, 8                  ; ah can't be reached with a rex prefix
    xor r9b, [rcx]
    shl eax, 8
    xor ax, [r8+r9*2]
    movzx eax, ax
    inc rcx
    jmp .b
.out:
    ret

; rcx = int16 mono samples, rdx = how many, r8d = sample rate, r9 = FE.
; a whole .flac into FE_DATA (mem_alloc'd), FE_LEN bytes
global flac_encode
flac_encode:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 104
    ; locals: FE +32, rate +40, samples left +48, next sample +56, frame +64,
    ; smallest frame +72, biggest +80, this frame's start +88, first sample +96
    mov [rsp+32], r9
    mov eax, r8d
    mov [rsp+40], rax
    mov [rsp+48], rdx
    mov [rsp+56], rcx
    mov [rsp+96], rcx
    mov rbx, rdx                ; all the samples, for STREAMINFO and the md5
    mov ecx, SCRATCH
    call mem_alloc
    mov r12, rax
    ; worst case: verbatim everywhere plus frame headers
    mov rcx, rbx
    shr rcx, 12
    inc rcx
    shl rcx, 6
    lea rcx, [rcx+rbx*2+128]
    call mem_alloc
    mov r13, rax                ; the file
    mov rdx, [rsp+32]
    mov [rdx+FE_DATA], rax
    lea rdi, [rax+42]           ; frames go after "fLaC" and STREAMINFO
    xor r14d, r14d
    xor r15d, r15d
    mov qword [rsp+64], 0
    mov qword [rsp+72], -1
    mov qword [rsp+80], 0
.frame:
    mov rax, [rsp+48]
    test rax, rax
    jz .frames
    mov ebp, BLOCK
    cmp rax, rbp
    cmovb ebp, eax
    mov [rsp+88], rdi
    putc 16, 0xfff8             ; sync, fixed block size
    mov edx, 12                 ; 4096
    cmp ebp, BLOCK
    je .bs
    mov edx, 7                  ; or 16 bits of size - 1 after the header
.bs:
    mov ecx, 4
    call put
    xor edx, edx                ; rate from STREAMINFO, unless it's 16 kHz
    cmp dword [rsp+40], 16000
    jne .rate
    mov edx, 5
.rate:
    mov ecx, 4
    call put
    putc 4, 0                   ; mono
    putc 3, 4                   ; 16 bits
    putc 1, 0
    mov edx, [rsp+64]
    call utf8
    cmp ebp, BLOCK
    je .crc8
    lea edx, [rbp-1]
    mov ecx, 16
    call put
.crc8:
    mov rcx, [rsp+88]
    mov rdx, rdi
    call crc8
    mov edx, eax
    mov ecx, 8
    call put
    mov rsi, [rsp+56]
    call subframe_enc
    test r15d, r15d             ; pad to a byte
    jz .padded
    mov ecx, 8
    sub ecx, r15d
    xor edx, edx
    call put
.padded:
    mov rcx, [rsp+88]
    mov rdx, rdi
    call crc16
    mov edx, eax
    mov ecx, 16
    call put
    mov rax, rdi
    sub rax, [rsp+88]
    cmp rax, [rsp+72]
    jae .notsmall
    mov [rsp+72], rax
.notsmall:
    cmp rax, [rsp+80]
    jbe .notbig
    mov [rsp+80], rax
.notbig:
    add [rsp+56], rbp
    add [rsp+56], rbp           ; int16
    sub [rsp+48], rbp
    inc qword [rsp+64]
    jmp .frame

.frames:
    ; "fLaC", STREAMINFO (last metadata block, 34 bytes)
    mov dword [r13], 'fLaC'
    mov dword [r13+4], 0x22000080
    mov word [r13+8], 0x0010    ; min block size 4096, big endian
    mov word [r13+10], 0x0010
    mov rax, [rsp+72]
    cmp rax, -1
    jne .minf
    xor eax, eax
.minf:
    bswap eax
    shr eax, 8
    mov [r13+12], ax            ; 24 bits big endian, written as 3 bytes
    shr eax, 16
    mov [r13+14], al
    mov rax, [rsp+80]
    bswap eax
    shr eax, 8
    mov [r13+15], ax
    shr eax, 16
    mov [r13+17], al
    ; rate 20 bits, channels-1 3, bits-1 5, samples 36
    mov rax, [rsp+40]
    shl rax, 44
    mov rcx, 15
    shl rcx, 36                 ; mono (0), 16 bits (15)
    or rax, rcx
    mov rcx, rbx
    mov rdx, 0xfffffffff
    and rcx, rdx
    or rax, rcx
    bswap rax
    mov [r13+18], rax
    mov rcx, [rsp+96]
    lea rdx, [rbx+rbx]
    lea r8, [r13+26]
    call md5
    mov rdx, [rsp+32]
    sub rdi, r13
    mov [rdx+FE_LEN], rdi
    mov rcx, r12
    call mem_free
    add rsp, 104
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret
