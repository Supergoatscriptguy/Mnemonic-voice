; flac: the stream header, and decoding to int16. handles what our data has: up to
; 16 bits a sample, mono or independent channels, every subframe type. the md5 in
; STREAMINFO covers the decoded samples, and for 16 bit those are exactly the
; int16 bytes we write, so checking it is one md5 of the output
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

section .rdata
align 4
md5k    dd 0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501
        dd 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821
        dd 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8
        dd 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a
        dd 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70
        dd 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665
        dd 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1
        dd 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
md5iv   dd 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476
; frame header sample size codes, 0 = from STREAMINFO, -1 = reserved
sizes   dd 0, 8, 12, -1, 16, 20, 24, 32

section .text

; rcx = bytes, rdx = size, r8 = FI to fill. eax = 1 if it starts with a flac
; stream header. everything in it is big endian
global flac_info
flac_info:
    push rbx
    push rsi
    push rdi
    mov rsi, rcx
    lea rdi, [rcx+rdx]
    mov rbx, r8
    xor eax, eax
    mov [rbx+FI_BLOCKS], eax
    mov [rbx+FI_FRAMES], rax
    cmp rdx, 42
    jb .no
    cmp dword [rsi], 'fLaC'
    jne .no
    add rsi, 4
.block:
    lea rax, [rsi+4]
    cmp rax, rdi
    ja .no
    mov eax, [rsi]
    bswap eax
    mov ecx, eax
    shr ecx, 24
    and ecx, 127                ; type
    mov edx, 1
    shl edx, cl
    or [rbx+FI_BLOCKS], edx
    and eax, 0xffffff           ; length
    add rsi, 4
    test ecx, ecx
    jnz .next
    cmp eax, 34
    jb .no
    ; STREAMINFO
    movzx edx, word [rsi]
    xchg dl, dh
    mov [rbx+FI_MINBS], edx
    movzx edx, word [rsi+2]
    xchg dl, dh
    mov [rbx+FI_MAXBS], edx
    mov edx, [rsi+3]            ; min frame size is bytes 4..6
    bswap edx
    and edx, 0xffffff
    mov [rbx+FI_MINFR], edx
    mov edx, [rsi+6]
    bswap edx
    and edx, 0xffffff
    mov [rbx+FI_MAXFR], edx
    ; bytes 10..17: rate 20 bits, channels-1 3, bits-1 5, total samples 36
    mov rdx, [rsi+10]
    bswap rdx
    mov rcx, rdx
    shr rcx, 44
    mov [rbx+FI_RATE], ecx
    mov rcx, rdx
    shr rcx, 41
    and ecx, 7
    inc ecx
    mov [rbx+FI_CH], ecx
    mov rcx, rdx
    shr rcx, 36
    and ecx, 31
    inc ecx
    mov [rbx+FI_BITS], ecx
    mov rcx, 0xfffffffff
    and rdx, rcx
    mov [rbx+FI_SAMPLES], rdx
    mov rdx, [rsi+18]
    mov [rbx+FI_MD5], rdx
    mov rdx, [rsi+26]
    mov [rbx+FI_MD5+8], rdx
.next:
    test byte [rsi-4], 0x80     ; last block
    lea rsi, [rsi+rax]
    jz .block
    cmp rsi, rdi
    ja .no
    mov [rbx+FI_FRAMES], rsi
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    pop rdi
    pop rsi
    pop rbx
    ret

; ---- md5 (rfc 1321)

; one of the 64 steps: A = B + rotl(A + f(B, C, D) + M[g] + K[i], s), then the
; names rotate so the next step's A is this one's D
%macro md5step 1
    %assign i %1
    %assign r i / 16
    %assign j i % 4
    %if r == 0
        mov r10d, C
        xor r10d, D
        and r10d, B
        xor r10d, D
        %assign g i
        %if j == 0
            %assign s 7
        %elif j == 1
            %assign s 12
        %elif j == 2
            %assign s 17
        %else
            %assign s 22
        %endif
    %elif r == 1
        mov r10d, B
        xor r10d, C
        and r10d, D
        xor r10d, C
        %assign g (5*i + 1) % 16
        %if j == 0
            %assign s 5
        %elif j == 1
            %assign s 9
        %elif j == 2
            %assign s 14
        %else
            %assign s 20
        %endif
    %elif r == 2
        mov r10d, B
        xor r10d, C
        xor r10d, D
        %assign g (3*i + 5) % 16
        %if j == 0
            %assign s 4
        %elif j == 1
            %assign s 11
        %elif j == 2
            %assign s 16
        %else
            %assign s 23
        %endif
    %else
        mov r10d, D
        not r10d
        or r10d, B
        xor r10d, C
        %assign g (7*i) % 16
        %if j == 0
            %assign s 6
        %elif j == 1
            %assign s 10
        %elif j == 2
            %assign s 15
        %else
            %assign s 21
        %endif
    %endif
    add A, r10d
    add A, [rdx + g*4]
    add A, [md5k + i*4]
    rol A, s
    add A, B
    %xdefine T D
    %xdefine D C
    %xdefine C B
    %xdefine B A
    %xdefine A T
%endmacro

; rcx = state (4 dwords), rdx = 64 bytes
md5_block:
    push rbx
    push rsi
    push rdi
    mov eax, [rcx]
    mov ebx, [rcx+4]
    mov esi, [rcx+8]
    mov edi, [rcx+12]
    %xdefine A eax
    %xdefine B ebx
    %xdefine C esi
    %xdefine D edi
    %assign k 0
    %rep 64
        md5step k
        %assign k k+1
    %endrep
    add [rcx], eax
    add [rcx+4], ebx
    add [rcx+8], esi
    add [rcx+12], edi
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = bytes, rdx = how many, r8 = 16 bytes for the digest
global md5
md5:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 168                ; a 128 byte tail buffer at +32
    mov rsi, rcx
    mov rbx, rdx
    mov r12, r8
    mov rax, [md5iv]
    mov [r12], rax
    mov rax, [md5iv+8]
    mov [r12+8], rax
    mov rdi, rbx
.whole:
    cmp rdi, 64
    jb .tail
    mov rcx, r12
    mov rdx, rsi
    call md5_block
    add rsi, 64
    sub rdi, 64
    jmp .whole
.tail:
    ; what's left, a 1 bit, zeros, the length in bits: one block or two
    xor eax, eax
.zero:
    mov qword [rsp+32+rax], 0
    add eax, 8
    cmp eax, 128
    jb .zero
    mov rcx, rdi
    mov rdx, rdi
    lea rdi, [rsp+32]
    rep movsb
    mov rdi, rdx
    mov byte [rsp+32+rdi], 0x80
    lea rax, [rbx*8]
    cmp rdi, 56
    jae .two
    mov [rsp+32+56], rax
    mov rcx, r12
    lea rdx, [rsp+32]
    call md5_block
    jmp .out
.two:
    mov [rsp+32+120], rax
    mov rcx, r12
    lea rdx, [rsp+32]
    call md5_block
    mov rcx, r12
    lea rdx, [rsp+96]
    call md5_block
.out:
    add rsp, 168
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ---- frames. the bit reader is three registers shared by everything below:
; rsi = the stream, r12 = bit position, r13 = its length in bytes

; rax = byte index. the 8 bytes there, big endian, zeros past the end. keeps rcx, rdx
fpeek_slow:
    push rcx
    push rdx
    mov rdx, rax
    xor eax, eax
    mov ecx, 8
.b:
    shl rax, 8
    cmp rdx, r13
    jae .z
    movzx r8d, byte [rsi+rdx]
    or rax, r8
.z:
    inc rdx
    dec ecx
    jnz .b
    pop rdx
    pop rcx
    ret

; rax = the next 64 bits, msb first, at least 57 of them real. trashes rcx, rdx, r8
%macro fpeek 0
    mov rax, r12
    shr rax, 3
    lea rdx, [rax+8]
    cmp rdx, r13
    ja %%slow
    mov rax, [rsi+rax]
    bswap rax
    jmp %%got
%%slow:
    call fpeek_slow
%%got:
    mov ecx, r12d
    and ecx, 7
    shl rax, cl
%endmacro

; ecx = n (0-32). rax = the next n bits, unsigned. trashes rcx, rdx, r8
ubits:
    test ecx, ecx
    jz .zero
    mov r9d, ecx
    fpeek
    mov ecx, r9d
    add r12, rcx
    neg ecx
    add ecx, 64
    shr rax, cl
    ret
.zero:
    xor eax, eax
    ret

; ecx = n (0-32). rax = the next n bits, sign extended. trashes rcx, rdx, r8, r9
sbits:
    test ecx, ecx
    jz .zero
    mov r9d, ecx
    fpeek
    mov ecx, r9d
    add r12, rcx
    neg ecx
    add ecx, 64
    sar rax, cl
    ret
.zero:
    xor eax, eax
    ret

; the rice coded residual of a subframe. rdi = where it goes (past the warmup),
; r8d = block size, r9d = predictor order, rbx = FD. eax = 1, 0 if it's broken
residual:
    push rbp
    push r14
    push r15
    mov ebp, r8d                ; block size
    mov r15d, r9d               ; order
    mov ecx, 2
    call ubits
    cmp eax, 1
    ja .bad
    mov r14d, 4                 ; rice parameter bits
    jb .order
    inc qword [rbx+FD_STATS+8*8]
    mov r14d, 5
.order:
    mov ecx, 4
    call ubits
    mov ecx, eax
    mov r11d, 1
    shl r11d, cl                ; partitions
    mov eax, ebp
    shr eax, cl                 ; samples in each
    mov edx, eax
    shl edx, cl
    cmp edx, ebp
    jne .bad                    ; the block has to split evenly
    mov ebp, eax
    mov r10d, eax
    sub r10d, r15d              ; the first one is short by the warmup
    jb .bad
.part:
    mov ecx, r14d
    call ubits
    mov r15d, eax               ; rice parameter
    mov eax, 1
    mov ecx, r14d
    shl eax, cl
    dec eax
    cmp r15d, eax
    je .escape
    test r10d, r10d
    jz .nextpart
.rs:
    fpeek
    lzcnt r8, rax               ; q, the unary part
    lea edx, [r8+r15+1]
    cmp edx, 57
    ja .longq
    lea ecx, [r8+1]
    shl rax, cl
    mov ecx, r15d
    rol rax, cl
    bzhi rax, rax, rcx          ; the k low bits
    add r12, rdx
    shl r8, cl
    or rax, r8
.zig:
    mov edx, eax
    shr eax, 1
    and edx, 1
    neg edx
    xor eax, edx
    mov [rdi], eax
    add rdi, 4
    dec r10d
    jnz .rs
    jmp .nextpart
.longq:
    ; a long run of zeros, counted 57 bits at a time
    inc qword [rbx+FD_STATS+13*8]
    xor r9d, r9d
.zeros:
    fpeek
    test rax, rax
    jnz .one
    add r9, 57
    add r12, 57
    cmp r9, 1 << 20
    ja .bad
    jmp .zeros
.one:
    lzcnt rcx, rax
    add r9, rcx
    lea r12, [r12+rcx+1]
    push r9
    mov ecx, r15d
    call ubits
    pop r9
    mov ecx, r15d
    shl r9, cl
    or rax, r9
    jmp .zig
.escape:
    inc qword [rbx+FD_STATS+9*8]
    mov ecx, 5
    call ubits
    mov r15d, eax               ; raw bits a sample
    test r10d, r10d
    jz .nextpart
.raw:
    mov ecx, r15d
    call sbits
    mov [rdi], eax
    add rdi, 4
    dec r10d
    jnz .raw
.nextpart:
    mov r10d, ebp
    dec r11d
    jnz .part
    mov eax, 1
    jmp .out
.bad:
    xor eax, eax
.out:
    pop r15
    pop r14
    pop rbp
    ret

; warmup samples for a fixed or lpc subframe: [rsp+160] of them into r15
%macro warmup 0
    mov r10d, [rsp+160]
    mov rdi, r15
    test r10d, r10d
    jz %%done
%%w:
    mov ecx, r14d
    call sbits
    mov [rdi], eax
    add rdi, 4
    dec r10d
    jnz %%w
%%done:
%endmacro

; one subframe. rdi = int32 out, r8d = block size, r9d = bits a sample, rbx = FD.
; eax = 1, 0 if it's broken or something we don't handle
subframe:
    push rbp
    push r14
    push r15
    sub rsp, 176
    ; locals: lpc coefficients +32 (32 of them), order +160, precision +164,
    ; shift +168, wasted bits +172
    mov ebp, r8d
    mov r14d, r9d
    mov r15, rdi
    mov dword [rsp+172], 0
    mov ecx, 8
    call ubits
    test eax, 0x80              ; the padding bit has to be 0
    jnz .bad
    mov r10d, eax
    shr r10d, 1
    and r10d, 63                ; type
    test eax, 1
    jz .typed
    ; wasted bits: k = 1 + the zeros before the next 1
    inc qword [rbx+FD_STATS+10*8]
.w:
    inc dword [rsp+172]
    mov ecx, 1
    call ubits
    test eax, eax
    jz .w
    mov eax, [rsp+172]
    cmp eax, r14d
    jae .bad
    sub r14d, eax
.typed:
    test r10d, r10d
    jz .constant
    cmp r10d, 1
    je .verbatim
    cmp r10d, 8
    jb .bad
    cmp r10d, 12
    jbe .fixed
    cmp r10d, 32
    jae .lpc
    jmp .bad

.constant:
    inc qword [rbx+FD_STATS]
    mov ecx, r14d
    call sbits
    mov rdi, r15
    mov ecx, ebp
    rep stosd
    jmp .shift
.verbatim:
    inc qword [rbx+FD_STATS+8]
    mov rdi, r15
    mov r10d, ebp
.v:
    mov ecx, r14d
    call sbits
    mov [rdi], eax
    add rdi, 4
    dec r10d
    jnz .v
    jmp .shift

.fixed:
    sub r10d, 8                 ; order 0-4
    mov [rsp+160], r10d
    inc qword [rbx+FD_STATS+16+r10*8]
    warmup
    mov r8d, ebp
    mov r9d, [rsp+160]
    call residual
    test eax, eax
    jz .bad
    mov ecx, [rsp+160]
    lea rdi, [r15+rcx*4]
    mov edx, ebp
    sub edx, ecx
    jz .shift
    cmp ecx, 1
    jb .shift
    je .f1
    cmp ecx, 3
    jb .f2
    je .f3
.f4:
    mov eax, [rdi-4]            ; 4 s1 - 6 s2 + 4 s3 - s4
    add eax, [rdi-12]
    shl eax, 2
    mov r8d, [rdi-8]
    imul r8d, 6
    sub eax, r8d
    sub eax, [rdi-16]
    add [rdi], eax
    add rdi, 4
    dec edx
    jnz .f4
    jmp .shift
.f3:
    mov eax, [rdi-4]            ; 3 (s1 - s2) + s3
    sub eax, [rdi-8]
    imul eax, 3
    add eax, [rdi-12]
    add [rdi], eax
    add rdi, 4
    dec edx
    jnz .f3
    jmp .shift
.f2:
    mov eax, [rdi-4]            ; 2 s1 - s2
    add eax, eax
    sub eax, [rdi-8]
    add [rdi], eax
    add rdi, 4
    dec edx
    jnz .f2
    jmp .shift
.f1:
    mov eax, [rdi-4]
    add [rdi], eax
    add rdi, 4
    dec edx
    jnz .f1
    jmp .shift

.lpc:
    sub r10d, 31                ; order 1-32
    mov [rsp+160], r10d
    inc qword [rbx+FD_STATS+7*8]
    cmp r10, [rbx+FD_STATS+12*8]
    jbe .lmax
    mov [rbx+FD_STATS+12*8], r10
.lmax:
    warmup
    mov ecx, 4
    call ubits
    cmp eax, 15
    je .bad
    inc eax
    mov [rsp+164], eax          ; coefficient precision
    mov ecx, 5
    call sbits
    test eax, eax
    js .bad                     ; negative shifts aren't allowed
    mov [rsp+168], eax
    xor r10d, r10d
.coef:
    mov ecx, [rsp+164]
    call sbits
    mov [rsp+32+r10*4], eax
    inc r10d
    cmp r10d, [rsp+160]
    jb .coef
    mov r8d, ebp
    mov r9d, [rsp+160]
    call residual
    test eax, eax
    jz .bad
    ; s[i] += (sum of c[j] s[i-1-j]) >> shift, summed in 64 bits
    mov r11d, [rsp+160]
    lea rdi, [r15+r11*4]
    mov edx, ebp
    sub edx, r11d
    jz .shift
    mov ecx, [rsp+168]
.ls:
    xor r9d, r9d
    mov r10, rdi
    xor r8d, r8d
.lj:
    sub r10, 4
    movsxd rax, dword [rsp+32+r8*4]
    movsxd r14, dword [r10]     ; r14 (the sample width) isn't needed any more
    imul rax, r14
    add r9, rax
    inc r8d
    cmp r8d, r11d
    jb .lj
    sar r9, cl
    add [rdi], r9d
    add rdi, 4
    dec edx
    jnz .ls

.shift:
    mov ecx, [rsp+172]          ; wasted bits go back on
    test ecx, ecx
    jz .ok
    mov rdi, r15
    mov edx, ebp
.sh:
    shl dword [rdi], cl
    add rdi, 4
    dec edx
    jnz .sh
.ok:
    mov eax, 1
    jmp .out
.bad:
    xor eax, eax
.out:
    add rsp, 176
    pop r15
    pop r14
    pop rbp
    ret

; rcx = bytes, rdx = size, r8 = FD. decodes a whole flac stream (9-16 bits, mono or
; independent channels) to int16, interleaved, in FD_PCM (mem_alloc'd). eax = 1 if it
; decoded, and then FD_MD5 says whether the md5 matched. 0 if it's broken or not
; something we handle
global flac_decode
flac_decode:
    push rbx
    push rsi
    push rdi
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 88
    ; locals: channel scratch +32, block size +40, channel assignment +48, digest +56
    mov rbx, r8
    mov rsi, rcx
    mov r13, rdx
    lea rdi, [rbx+FD_PCM]
    xor eax, eax
    mov ecx, (FD_SIZE - FD_PCM) / 8
    rep stosq
    mov qword [rsp+32], 0
    mov rcx, rsi
    mov rdx, r13
    mov r8, rbx
    call flac_info
    test eax, eax
    jz .bad
    mov eax, [rbx+FI_BITS]
    cmp eax, 9
    jb .bad
    cmp eax, 16
    ja .bad
    cmp dword [rbx+FI_CH], 8
    ja .bad
    mov rax, [rbx+FI_SAMPLES]
    test rax, rax
    jz .bad
    mov ecx, [rbx+FI_CH]
    imul rcx, rax
    lea rcx, [rcx*2+64]
    call mem_alloc
    mov [rbx+FD_PCM], rax
    mov ecx, [rbx+FI_CH]
    shl rcx, 18                 ; 65536 int32 a channel, the biggest block there is
    call mem_alloc
    mov [rsp+32], rax
    mov r12, [rbx+FI_FRAMES]
    sub r12, rsi
    shl r12, 3
    xor r14d, r14d              ; samples (a channel) so far
.frame:
    cmp r14, [rbx+FI_SAMPLES]
    jae .done
    mov rax, r12
    shr rax, 3
    add rax, 6
    cmp rax, r13
    ja .done                    ; the stream ran out: the md5 will say so
    inc qword [rbx+FD_STATS+11*8]
    mov ecx, 16
    call ubits
    and eax, 0xfffe
    cmp eax, 0xfff8             ; sync
    jne .bad
    mov ecx, 4
    call ubits
    mov ebp, eax                ; block size code
    mov ecx, 4
    call ubits
    mov r15d, eax               ; sample rate code
    mov ecx, 4
    call ubits
    mov [rsp+48], eax           ; channel assignment
    mov ecx, 3
    call ubits
    mov edi, eax                ; sample size code
    mov ecx, 1
    call ubits
    ; the frame number, coded like utf-8: n leading ones = n-1 more bytes
    mov ecx, 8
    call ubits
    mov ecx, eax
    not ecx
    shl ecx, 24
    lzcnt ecx, ecx
    cmp ecx, 1
    je .bad
    jb .numbered
    dec ecx
    shl ecx, 3
    add r12, rcx
.numbered:
    test ebp, ebp
    jz .bad
    cmp ebp, 1
    jne .bs2
    mov eax, 192
    jmp .bs
.bs2:
    cmp ebp, 5
    ja .bs6
    lea ecx, [rbp-2]
    mov eax, 576
    shl eax, cl
    jmp .bs
.bs6:
    cmp ebp, 6
    jne .bs7
    mov ecx, 8
    call ubits
    inc eax
    jmp .bs
.bs7:
    cmp ebp, 7
    jne .bs8
    mov ecx, 16
    call ubits
    inc eax
    jmp .bs
.bs8:
    lea ecx, [rbp-8]
    mov eax, 256
    shl eax, cl
.bs:
    mov [rsp+40], eax
    cmp r15d, 12                ; rates that don't fit the code come after the header
    jb .rated
    je .r8
    cmp r15d, 15
    je .bad
    add r12, 16
    jmp .rated
.r8:
    add r12, 8
.rated:
    lea rax, [sizes]
    mov eax, [rax+rdi*4]
    cmp eax, -1
    je .bad
    test eax, eax
    jz .sized
    cmp eax, [rbx+FI_BITS]
    jne .bad
.sized:
    add r12, 8                  ; crc-8
    mov eax, [rsp+48]
    cmp eax, 7
    ja .bad                     ; left/side, side/right, mid/side: not in our data
    inc eax
    cmp eax, [rbx+FI_CH]
    jne .bad
    mov eax, [rsp+40]
    add rax, r14
    cmp rax, [rbx+FI_SAMPLES]
    ja .bad
    xor ebp, ebp
.chan:
    mov rdi, [rsp+32]
    mov eax, ebp
    shl rax, 18
    add rdi, rax
    mov r8d, [rsp+40]
    mov r9d, [rbx+FI_BITS]
    call subframe
    test eax, eax
    jz .bad
    inc ebp
    cmp ebp, [rbx+FI_CH]
    jb .chan
    add r12, 7                  ; pad to a byte, then the crc-16
    and r12, -8
    add r12, 16
    ; int16, interleaved
    mov r8d, [rbx+FI_CH]
    mov rdi, r14
    imul rdi, r8
    add rdi, rdi
    add rdi, [rbx+FD_PCM]
    xor ebp, ebp
.ilv:
    mov rcx, [rsp+32]
    mov eax, ebp
    shl rax, 18
    add rcx, rax
    lea rdx, [rdi+rbp*2]
    mov r9d, [rsp+40]
.s:
    mov eax, [rcx]
    mov [rdx], ax
    add rcx, 4
    lea rdx, [rdx+r8*2]
    dec r9d
    jnz .s
    inc ebp
    cmp ebp, r8d
    jb .ilv
    mov eax, [rsp+40]
    add r14, rax
    jmp .frame

.done:
    mov [rbx+FD_N], r14
    mov dword [rbx+FD_MD5], 2
    mov rax, [rbx+FI_MD5]
    or rax, [rbx+FI_MD5+8]
    jz .free
    mov rcx, [rbx+FD_PCM]
    mov edx, [rbx+FI_CH]
    imul rdx, r14
    add rdx, rdx
    lea r8, [rsp+56]
    call md5
    xor eax, eax
    mov rcx, [rsp+56]
    cmp rcx, [rbx+FI_MD5]
    jne .md5
    mov rcx, [rsp+64]
    cmp rcx, [rbx+FI_MD5+8]
    jne .md5
    mov eax, 1
.md5:
    mov [rbx+FD_MD5], eax
.free:
    mov rcx, [rsp+32]
    call mem_free
    mov eax, 1
    jmp .out
.bad:
    mov rcx, [rsp+32]
    test rcx, rcx
    jz .noscratch
    call mem_free
.noscratch:
    mov rcx, [rbx+FD_PCM]
    test rcx, rcx
    jz .nopcm
    call mem_free
    mov qword [rbx+FD_PCM], 0
.nopcm:
    xor eax, eax
.out:
    add rsp, 88
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rdi
    pop rsi
    pop rbx
    ret
