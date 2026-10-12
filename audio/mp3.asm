; mp3: mpeg-1, 2 and 2.5 layer iii. mp3_info walks any file's frame headers,
; mp3_decode decodes mpeg-1 mono (all of common voice) to f32. iso 11172-3
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

; one granule's side info
GR_P23      equ 0               ; part2_3_length, bits
GR_BIGV     equ 4
GR_GAIN     equ 8
GR_SCOMP    equ 12
GR_WSF      equ 16              ; window switching
GR_BTYPE    equ 20
GR_MIXED    equ 24
GR_TSEL     equ 28              ; [3]
GR_SBG      equ 40              ; [3] subblock gains
GR_R0       equ 52
GR_R1       equ 56
GR_PRE      equ 60
GR_SSCALE   equ 64
GR_C1SEL    equ 68
GR_SIZE     equ 72

; the decoder's state (M3_SIZE in audio.inc), one per thread
M3_NRES     equ 0               ; u64 bytes in the bit reservoir
M3_VOFF     equ 8               ; u64 where the synthesis ring starts
M3_BP       equ 16              ; u64 bit position
M3_BUF      equ 24              ; -> what the bits come from
M3_MDB      equ 32              ; u32 main_data_begin
M3_SCFSI    equ 36              ; u32
M3_END      equ 40              ; u64 bit position where the granule's data ends
M3_GR       equ 64              ; [2] GR
M3_SCL      equ 256             ; u8 [22] long scalefactors, granule 1 can reuse them
M3_SCS      equ 288             ; u8 [13][3] short ones
M3_IS       equ 384             ; i32 [576] the huffman values
M3_XR       equ 2752            ; f32 [576] requantized
M3_TMP      equ 5056            ; f32 [576] for reordering
M3_OVL      equ 7424            ; f32 [32][18] imdct overlap
M3_V        equ 9728            ; f32 [1024] synthesis ring
M3_OUT      equ 13824           ; f32 [18][32] subband samples, time first
M3_Z        equ 16128           ; f32 [40] one imdct
M3_ZS       equ 16288           ; f32 [16] one short window
M3_RES      equ 16384           ; the reservoir's bytes
RESCAP      equ 4096

section .rdata
br1     dw 0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0
br2     dw 0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0
sr1     dd 44100, 48000, 32000, 0
slen1   db 0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4
slen2   db 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3
pretab  db 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0
grp     db 0, 6, 11, 16, 21     ; scfsi's band groups
align 2
; scalefactor band edges in lines, at 44.1, 48 and 32 kHz
sfbl    dw 0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576
        dw 0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576
        dw 0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576
sfbs    dw 0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136, 192
        dw 0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126, 192
        dw 0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180, 192
align 4
; each huffman table: where it starts in htab, its tree's length, linbits
hidx    dd 0, 0, 0,  0, 7, 0,  7, 17, 0,  24, 17, 0,  0, 0, 0,  41, 31, 0,  72, 31, 0
        dd 103, 71, 0,  174, 71, 0,  245, 71, 0,  316, 127, 0,  443, 127, 0,  570, 127, 0
        dd 697, 511, 0,  0, 0, 0,  1208, 511, 0
        dd 1719, 511, 1,  1719, 511, 2,  1719, 511, 3,  1719, 511, 4
        dd 1719, 511, 6,  1719, 511, 8,  1719, 511, 10,  1719, 511, 13
        dd 2230, 512, 4,  2230, 512, 5,  2230, 512, 6,  2230, 512, 7
        dd 2230, 512, 8,  2230, 512, 9,  2230, 512, 11,  2230, 512, 13
        dd 2742, 31, 0,  2773, 31, 0
qtab    dd 1.0, 1.189207115002721, 1.414213562373095, 1.681792830507429  ; 2^(i/4)
align 8
c_ci    dq -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037
c_pi    dq 3.141592653589793
c_43    dq 1.3333333333333333
c_one   dq 1.0
align 16
c_sign  dd 0x80000000, 0x80000000, 0x80000000, 0x80000000
%include "audio/mp3tab.inc"

section .bss
alignb 32
pow43   resd 8208               ; |x|^(4/3)
cosl    resd 18 * 40            ; long imdct, [k][i], i padded to 40
coss    resd 6 * 16             ; short imdct with its window, [k][i]
win     resd 4 * 40             ; long windows by block type
nt      resd 32 * 64            ; synthesis matrixing, [k][i]
acs     resd 8                  ; alias reduction
aca     resd 8
inited  resd 1

section .text

; ecx = a frame header, big endian order (bswapped). eax = the frame's bytes, 0 if
; it isn't a layer iii header we can use, edx = sample rate, r8d = samples a
; channel, r9d = kbps. r10d = version (1, 2, 25)
global mp3_head
mp3_head:
    mov eax, ecx
    shr eax, 21
    cmp eax, 0x7ff
    jne .no
    mov eax, ecx
    shr eax, 17
    and eax, 3
    cmp eax, 1                  ; layer iii
    jne .no
    mov eax, ecx
    shr eax, 19
    and eax, 3
    cmp eax, 1
    je .no
    mov r10d, 1
    cmp eax, 3
    je .v
    mov r10d, 2
    cmp eax, 2
    je .v
    mov r10d, 25
.v:
    mov eax, ecx
    shr eax, 10
    and eax, 3
    cmp eax, 3
    je .no
    lea rdx, [sr1]
    mov edx, [rdx+rax*4]
    cmp r10d, 1
    je .r
    shr edx, 1
    cmp r10d, 2
    je .r
    shr edx, 1
.r:
    mov eax, ecx
    shr eax, 12
    and eax, 15
    lea r8, [br1]
    cmp r10d, 1
    je .b
    lea r8, [br2]
.b:
    movzx r9d, word [r8+rax*2]
    test r9d, r9d
    jz .no                      ; free format or bad
    ; bytes: 144000 * kbps / rate (72000 past mpeg-1), plus the padding slot
    mov eax, 144000
    mov r8d, 1152
    cmp r10d, 1
    je .len
    mov eax, 72000
    mov r8d, 576
.len:
    push rdx
    imul eax, r9d
    mov r11d, edx
    xor edx, edx
    div r11d
    pop rdx
    bt ecx, 9
    adc eax, 0
    ret
.no:
    xor eax, eax
    ret

; rcx = the file's bytes, rdx = size, r8 = MI. walks every frame. eax = 1 if there
; were any
global mp3_info
mp3_info:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    mov rsi, rcx
    mov r12, rdx
    mov rbx, r8
    xor eax, eax
.z:
    mov qword [rbx+rax], 0
    add eax, 8
    cmp eax, MI_SIZE
    jb .z
    mov dword [rbx+MI_MINBR], 0xffff
    xor edi, edi
    ; an id3v2 tag first: "ID3", version, flags, size in 7-bit bytes
    cmp r12, 10
    jb .walk
    mov eax, [rsi]
    and eax, 0xffffff
    cmp eax, 'ID3'
    jne .walk
    xor eax, eax
    xor ecx, ecx
.syn:
    shl eax, 7
    movzx edx, byte [rsi+6+rcx]
    and edx, 0x7f
    or eax, edx
    inc ecx
    cmp ecx, 4
    jb .syn
    lea rdi, [rax+10]
    test byte [rsi+5], 0x10     ; a footer too
    jz .walk
    add rdi, 10
.walk:
    lea rax, [rdi+4]
    cmp rax, r12
    ja .end
    mov ecx, [rsi+rdi]
    bswap ecx
    mov r13d, ecx
    call mp3_head
    test eax, eax
    jz .lost
    lea rcx, [rdi+rax]
    cmp rcx, r12
    ja .end                     ; cut off at the end
    mov r14d, eax
    cmp qword [rbx+MI_FRAMES], 0
    jne .more
    ; the first frame says what the file is
    mov [rbx+MI_VER], r10d
    mov [rbx+MI_RATE], edx
    mov eax, r13d
    shr eax, 6
    and eax, 3
    mov [rbx+MI_MODE], eax
    mov eax, r13d
    shr eax, 16
    and eax, 1
    xor eax, 1
    mov [rbx+MI_CRC], eax
    ; a xing/info frame: its tag sits after the side info
    mov eax, 4
    add eax, [rbx+MI_CRC]
    add eax, [rbx+MI_CRC]
    mov ecx, 17                 ; side info: mpeg-1 mono 17, else 32; later 9 / 17
    cmp dword [rbx+MI_MODE], 3
    je .si1
    mov ecx, 32
.si1:
    cmp r10d, 1
    je .si2
    mov ecx, 9
    cmp dword [rbx+MI_MODE], 3
    je .si2
    mov ecx, 17
.si2:
    add eax, ecx
    add rax, rdi
    lea rcx, [rax+4]
    cmp rcx, r12
    ja .more
    mov ecx, [rsi+rax]
    cmp ecx, 'Xing'
    je .xing
    cmp ecx, 'Info'
    jne .more
.xing:
    mov dword [rbx+MI_XING], 1
.more:
    cmp edx, [rbx+MI_RATE]
    jne .mixed
    cmp r10d, [rbx+MI_VER]
    je .same
.mixed:
    mov dword [rbx+MI_MIXED], 1
.same:
    inc qword [rbx+MI_FRAMES]
    add [rbx+MI_SAMPLES], r8
    cmp r9d, [rbx+MI_MINBR]
    jae .mn
    mov [rbx+MI_MINBR], r9d
.mn:
    cmp r9d, [rbx+MI_MAXBR]
    jbe .mx
    mov [rbx+MI_MAXBR], r9d
.mx:
    mov eax, r13d
    shr eax, 6
    and eax, 3
    bts dword [rbx+MI_MODES], eax
    add rdi, r14
    jmp .walk
.lost:
    ; an id3v1 tag at the very end is fine, anything else is skipped over
    lea rax, [rdi+128]
    cmp rax, r12
    jne .skip
    mov eax, [rsi+rdi]
    and eax, 0xffffff
    cmp eax, 'TAG'
    jne .skip
    mov rdi, r12
    jmp .end
.skip:
    inc qword [rbx+MI_SKIP]
    inc rdi
    jmp .walk
.end:
    mov rax, r12
    sub rax, rdi
    mov [rbx+MI_TAIL], rax
    xor eax, eax
    cmp qword [rbx+MI_FRAMES], 0
    setne al
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ---- decoding

; ecx = n, edx = d: xmm0 = cos(pi n / d), and sin
cospi:
    sub rsp, 40
    cvtsi2sd xmm0, ecx
    mulsd xmm0, [c_pi]
    cvtsi2sd xmm1, edx
    divsd xmm0, xmm1
    call math_cos
    add rsp, 40
    ret
sinpi:
    sub rsp, 40
    cvtsi2sd xmm0, ecx
    mulsd xmm0, [c_pi]
    cvtsi2sd xmm1, edx
    divsd xmm0, xmm1
    call math_sin
    add rsp, 40
    ret

; the tables worked out from their formulas. once, before any decoding
global mp3_init
mp3_init:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    cmp dword [inited], 0
    jne .out
    ; |x|^(4/3)
    mov ebx, 1
.p:
    cvtsi2sd xmm0, ebx
    call math_log
    mulsd xmm0, [c_43]
    call math_exp
    cvtsd2ss xmm0, xmm0
    lea rax, [pow43]
    movss [rax+rbx*4], xmm0
    inc ebx
    cmp ebx, 8208
    jb .p
    ; long imdct: cos(pi/72 (2i + 19)(2k + 1)), the angle reduced mod 144 first
    xor esi, esi                ; k
.lk:
    xor edi, edi                ; i
.li:
    lea eax, [rdi*2+19]
    lea ecx, [rsi*2+1]
    imul eax, ecx
    xor edx, edx
    mov ecx, 144
    div ecx
    mov ecx, edx
    mov edx, 72
    call cospi
    cvtsd2ss xmm0, xmm0
    imul eax, esi, 40
    add eax, edi
    lea rcx, [cosl]
    movss [rcx+rax*4], xmm0
    inc edi
    cmp edi, 36
    jb .li
    inc esi
    cmp esi, 18
    jb .lk
    ; short imdct, its sine window folded in: cos(pi/24 (2i + 7)(2k + 1)) sin(pi/24 (2i + 1))
    xor esi, esi
.sk:
    xor edi, edi
.si:
    lea eax, [rdi*2+7]
    lea ecx, [rsi*2+1]
    imul eax, ecx
    xor edx, edx
    mov ecx, 48
    div ecx
    mov ecx, edx
    mov edx, 24
    call cospi
    movsd [rsp+32], xmm0
    lea ecx, [rdi*2+1]
    mov edx, 24
    call sinpi
    mulsd xmm0, [rsp+32]
    cvtsd2ss xmm0, xmm0
    imul eax, esi, 16
    add eax, edi
    lea rcx, [coss]
    movss [rcx+rax*4], xmm0
    inc edi
    cmp edi, 12
    jb .si
    inc esi
    cmp esi, 6
    jb .sk
    ; windows: 0 normal, 1 start, 3 stop (2, short, is in coss)
    xor edi, edi
.w:
    lea ecx, [rdi*2+1]
    mov edx, 72
    call sinpi
    cvtsd2ss xmm0, xmm0
    lea rbx, [win]
    movss [rbx+rdi*4], xmm0                 ; normal
    cmp edi, 18
    jae .w1
    movss [rbx+160+rdi*4], xmm0             ; start, rising half
    jmp .w3
.w1:
    movss [rbx+480+rdi*4], xmm0             ; stop, falling half
.w3:
    inc edi
    cmp edi, 36
    jb .w
    mov eax, __?float32?__(1.0)
    xor edi, edi
.w4:
    mov [rbx+160+72+rdi*4], eax             ; start 18-23 and stop 12-17 are 1
    mov [rbx+480+48+rdi*4], eax
    inc edi
    cmp edi, 6
    jb .w4
    xor edi, edi
.w5:
    lea ecx, [rdi*2+1]
    mov edx, 24
    call sinpi
    cvtsd2ss xmm0, xmm0
    lea rbx, [win]
    movss [rbx+480+24+rdi*4], xmm0          ; stop 6-11 rises
    mov eax, 5
    sub eax, edi
    lea rcx, [rbx+160+96]
    movss [rcx+rax*4], xmm0                 ; start 24-29 falls
    inc edi
    cmp edi, 6
    jb .w5
    ; matrixing: cos((16 + i)(2k + 1) pi / 64), reduced mod 128
    xor esi, esi
.nk:
    xor edi, edi
.ni:
    lea eax, [rdi+16]
    lea ecx, [rsi*2+1]
    imul eax, ecx
    and eax, 127
    mov ecx, eax
    mov edx, 64
    call cospi
    cvtsd2ss xmm0, xmm0
    mov eax, esi
    shl eax, 6
    add eax, edi
    lea rcx, [nt]
    movss [rcx+rax*4], xmm0
    inc edi
    cmp edi, 64
    jb .ni
    inc esi
    cmp esi, 32
    jb .nk
    ; alias reduction
    xor edi, edi
.a:
    lea rax, [c_ci]
    movsd xmm1, [rax+rdi*8]
    movapd xmm0, xmm1
    mulsd xmm0, xmm0
    addsd xmm0, [c_one]
    sqrtsd xmm0, xmm0
    movsd xmm2, [c_one]
    divsd xmm2, xmm0
    mulsd xmm1, xmm2
    cvtsd2ss xmm2, xmm2
    cvtsd2ss xmm1, xmm1
    lea rax, [acs]
    movss [rax+rdi*4], xmm2
    lea rax, [aca]
    movss [rax+rdi*4], xmm1
    inc edi
    cmp edi, 8
    jb .a
    mov dword [inited], 1
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; r15 = state. ecx = n (0-32): eax = the next n bits. only touches rax, rcx, rdx, r8
getbits:
    test ecx, ecx
    jz .zero
    mov r8, [r15+M3_BP]
    mov rdx, r8
    shr rdx, 3
    add rdx, [r15+M3_BUF]
    mov rax, [rdx]
    bswap rax
    and r8d, 7
    shlx rax, rax, r8
    add [r15+M3_BP], rcx
    neg ecx
    add ecx, 64
    shrx rax, rax, rcx
    ret
.zero:
    xor eax, eax
    ret

%macro gb 1
    mov ecx, %1
    call getbits
%endmacro

; r15 = state, ecx = table (0-33). under 32: eax = x, edx = y, with linbits and
; signs. 32 and 33: eax = the quad's bits, v highest, signs still to read.
; only touches rax, rcx, rdx, r8
huff:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    mov r13d, ecx
    lea rax, [hidx]
    lea rcx, [rcx+rcx*2]
    mov esi, [rax+rcx*4]
    mov edi, [rax+rcx*4+4]
    mov r12d, [rax+rcx*4+8]
    xor eax, eax
    xor edx, edx
    test edi, edi
    jz .out
    lea rbx, [htab]
    lea rbx, [rbx+rsi*2]
    xor esi, esi
.walk:
    movzx eax, word [rbx+rsi*2]
    test eax, 0xff00
    jz .leaf
    gb 1
    movzx edx, word [rbx+rsi*2]
    test eax, eax
    jz .left
.right:
    movzx eax, dl
    cmp eax, 250
    jb .r
    add esi, eax
    movzx edx, word [rbx+rsi*2]
    jmp .right
.r:
    add esi, eax
    jmp .next
.left:
    movzx eax, dh
    cmp eax, 250
    jb .l
    add esi, eax
    movzx edx, word [rbx+rsi*2]
    jmp .left
.l:
    add esi, eax
.next:
    cmp esi, edi
    jb .walk
    xor eax, eax                ; not a code: nothing
    xor edx, edx
    jmp .out
.leaf:
    cmp r13d, 32
    jae .quad
    mov edi, eax
    shr edi, 4
    and edi, 15
    mov esi, eax
    and esi, 15
    test r12d, r12d
    jz .xs
    cmp edi, 15
    jne .xs
    mov ecx, r12d
    call getbits
    add edi, eax
.xs:
    test edi, edi
    jz .y
    gb 1
    test eax, eax
    jz .y
    neg edi
.y:
    test r12d, r12d
    jz .ys
    cmp esi, 15
    jne .ys
    mov ecx, r12d
    call getbits
    add esi, eax
.ys:
    test esi, esi
    jz .xy
    gb 1
    test eax, eax
    jz .xy
    neg esi
.xy:
    mov eax, edi
    mov edx, esi
    jmp .out
.quad:
    and eax, 15
.out:
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; eax = q: xmm0 = 2^(q/4). touches rax, rcx
pow2q:
    mov ecx, eax
    sar ecx, 2
    and eax, 3
    add ecx, 127
    jle .zero
    cmp ecx, 254
    jbe .ok
    mov ecx, 254
.ok:
    shl ecx, 23
    movd xmm0, ecx
    lea rcx, [qtab]
    mulss xmm0, [rcx+rax*4]
    ret
.zero:
    xorps xmm0, xmm0
    ret

; r15 = state, ecx = first line, edx = lines, xmm0 = their gain. XR = sign IS |IS|^(4/3) gain
rq_lines:
    lea r8, [r15+M3_IS]
    lea r9, [r15+M3_XR]
    lea r10, [pow43]
.l:
    test edx, edx
    jz .done
    mov eax, [r8+rcx*4]
    mov r11d, eax
    test eax, eax
    jns .p
    neg eax
.p:
    cmp eax, 8207
    jb .ok
    mov eax, 8207
.ok:
    movss xmm1, [r10+rax*4]
    mulss xmm1, xmm0
    test r11d, r11d
    jns .s
    xorps xmm1, [c_sign]
.s:
    movss [r9+rcx*4], xmm1
    inc ecx
    dec edx
    jmp .l
.done:
    ret

; r15 = state, ebx = granule, r12 = where its 576 samples go, r13 = rate index.
; the bits are at M3_BP
granule:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 72
    imul r14d, ebx, GR_SIZE
    lea r14, [r15+r14+M3_GR]
    mov rax, [r15+M3_BP]
    mov ecx, [r14+GR_P23]
    add rax, rcx
    mov [r15+M3_END], rax
    imul eax, r13d, 46
    lea rcx, [sfbl]
    add rax, rcx
    mov [rsp+40], rax           ; this rate's long bands
    imul eax, r13d, 28
    lea rcx, [sfbs]
    add rax, rcx
    mov [rsp+48], rax           ;   and short ones
    ; kind: 0 long (block types 0, 1, 3), 1 short, 2 mixed
    xor eax, eax
    cmp dword [r14+GR_WSF], 0
    je .kind
    cmp dword [r14+GR_BTYPE], 2
    jne .kind
    mov eax, 1
    cmp dword [r14+GR_MIXED], 0
    je .kind
    mov eax, 2
.kind:
    mov [rsp+32], eax

    ; scalefactors
    mov eax, [r14+GR_SCOMP]
    lea rcx, [slen1]
    movzx esi, byte [rcx+rax]
    lea rcx, [slen2]
    movzx edi, byte [rcx+rax]
    cmp dword [rsp+32], 0
    jne .sshort
    ; long: four groups of bands, granule 1 can keep granule 0's
    xor r9d, r9d
.sg:
    test ebx, ebx
    jz .sread
    mov ecx, 3
    sub ecx, r9d
    bt dword [r15+M3_SCFSI], ecx
    jc .snext
.sread:
    lea rax, [grp]
    movzx r10d, byte [rax+r9]
    movzx r11d, byte [rax+r9+1]
    mov eax, esi
    cmp r9d, 2
    jb .sl
    mov eax, edi
.sl:
    mov [rsp+36], eax
.sb:
    cmp r10d, r11d
    jae .snext
    mov ecx, [rsp+36]
    call getbits
    mov [r15+r10+M3_SCL], al
    inc r10d
    jmp .sb
.snext:
    inc r9d
    cmp r9d, 4
    jb .sg
    mov byte [r15+M3_SCL+21], 0
    jmp .huff
.sshort:
    xor r10d, r10d
    cmp dword [rsp+32], 2
    jne .ss
.ml:
    mov ecx, esi                ; mixed: long bands 0-7, then short from band 3
    call getbits
    mov [r15+r10+M3_SCL], al
    inc r10d
    cmp r10d, 8
    jb .ml
    mov r10d, 3
.ss:
    cmp r10d, 12
    jae .ssd
    mov r11d, esi
    cmp r10d, 6
    jb .ss1
    mov r11d, edi
.ss1:
    imul r9d, r10d, 3
    mov ecx, r11d
    call getbits
    mov [r15+r9+M3_SCS], al
    mov ecx, r11d
    call getbits
    mov [r15+r9+M3_SCS+1], al
    mov ecx, r11d
    call getbits
    mov [r15+r9+M3_SCS+2], al
    inc r10d
    jmp .ss
.ssd:
    mov word [r15+M3_SCS+36], 0
    mov byte [r15+M3_SCS+38], 0

.huff:
    ; big values in up to three regions, each with its own table
    mov r10d, 36
    mov r11d, 576
    cmp dword [r14+GR_WSF], 0
    jne .reg
    mov rax, [rsp+40]
    mov ecx, [r14+GR_R0]
    inc ecx
    movzx r10d, word [rax+rcx*2]
    add ecx, [r14+GR_R1]
    inc ecx
    cmp ecx, 22
    jbe .r2
    mov ecx, 22
.r2:
    movzx r11d, word [rax+rcx*2]
.reg:
    mov esi, [r14+GR_BIGV]
    add esi, esi
    cmp esi, 576
    jbe .bv
    mov esi, 576
.bv:
    xor edi, edi
.big:
    cmp edi, esi
    jae .c1
    mov ecx, [r14+GR_TSEL]
    cmp edi, r10d
    jb .tab
    mov ecx, [r14+GR_TSEL+4]
    cmp edi, r11d
    jb .tab
    mov ecx, [r14+GR_TSEL+8]
.tab:
    call huff
    mov [r15+rdi*4+M3_IS], eax
    mov [r15+rdi*4+M3_IS+4], edx
    add edi, 2
    jmp .big
.c1:
    ; then quads of -1, 0, 1 until the granule's bits run out
    mov esi, [r14+GR_C1SEL]
    add esi, 32
.quad:
    cmp edi, 572
    ja .c1d
    mov rax, [r15+M3_BP]
    cmp rax, [r15+M3_END]
    jae .c1d
    mov ecx, esi
    call huff
    mov r9d, eax
    mov r10d, 3
.qv:
    xor r11d, r11d
    bt r9d, r10d
    jnc .qs
    mov r11d, 1
    gb 1
    test eax, eax
    jz .qs
    neg r11d
.qs:
    mov eax, 3
    sub eax, r10d
    add eax, edi
    mov [r15+rax*4+M3_IS], r11d
    dec r10d
    jns .qv
    add edi, 4
    jmp .quad
.c1d:
    mov rax, [r15+M3_BP]
    cmp rax, [r15+M3_END]
    jbe .zr
    sub edi, 4                  ; that quad took bits from the next granule
.zr:
    cmp edi, 576
    jae .rq
    mov dword [r15+rdi*4+M3_IS], 0
    inc edi
    jmp .zr

.rq:
    mov rax, [r15+M3_END]
    mov [r15+M3_BP], rax
    ; requantize: gain 2^((global - 210)/4), less the scalefactors (in quarter steps)
    mov esi, [r14+GR_GAIN]
    sub esi, 210
    mov edi, 2
    cmp dword [r14+GR_SSCALE], 0
    je .m
    mov edi, 4
.m:
    mov eax, 22
    cmp dword [rsp+32], 0
    je .nl
    xor eax, eax
    cmp dword [rsp+32], 2
    jne .nl
    mov eax, 8
.nl:
    mov [rsp+56], eax           ; long bands
    xor ebx, ebx
.rl:
    cmp ebx, [rsp+56]
    jae .rs
    movzx eax, byte [r15+rbx+M3_SCL]
    cmp dword [r14+GR_PRE], 0
    je .np
    lea rcx, [pretab]
    movzx ecx, byte [rcx+rbx]
    add eax, ecx
.np:
    imul eax, edi
    neg eax
    add eax, esi
    call pow2q
    mov rax, [rsp+40]
    movzx ecx, word [rax+rbx*2]
    movzx edx, word [rax+rbx*2+2]
    sub edx, ecx
    call rq_lines
    inc ebx
    jmp .rl
.rs:
    cmp dword [rsp+32], 0
    je .alias
    xor ebx, ebx                ; short bands, three windows each
    cmp dword [rsp+32], 2
    jne .sb2
    mov ebx, 3
.sb2:
    mov [rsp+60], ebx           ; the first short band (for reordering)
.rsb:
    cmp ebx, 13
    jae .reorder
    mov dword [rsp+36], 0       ; window
.rsw:
    mov ecx, [rsp+36]
    mov eax, [r14+rcx*4+GR_SBG]
    shl eax, 3
    imul ecx, ebx, 3
    add ecx, [rsp+36]
    movzx ecx, byte [r15+rcx+M3_SCS]
    imul ecx, edi
    add eax, ecx
    neg eax
    add eax, esi
    call pow2q
    mov rax, [rsp+48]
    movzx ecx, word [rax+rbx*2]
    movzx edx, word [rax+rbx*2+2]
    sub edx, ecx                ; width
    lea ecx, [rcx+rcx*2]
    mov eax, edx
    imul eax, [rsp+36]
    add ecx, eax
    call rq_lines
    inc dword [rsp+36]
    cmp dword [rsp+36], 3
    jb .rsw
    inc ebx
    jmp .rsb

.reorder:
    ; short bands come window by window: interleave them so line 3f + w is
    ; window w's frequency f, as the imdct wants
    mov ebx, [rsp+60]
.ro:
    cmp ebx, 13
    jae .rocopy
    mov rax, [rsp+48]
    movzx r8d, word [rax+rbx*2]
    movzx r9d, word [rax+rbx*2+2]
    sub r9d, r8d                ; width
    lea r8d, [r8+r8*2]          ; base
    xor r10d, r10d              ; window
.row:
    xor r11d, r11d              ; f
.rof:
    mov eax, r10d
    imul eax, r9d
    add eax, r11d
    add eax, r8d
    movss xmm0, [r15+rax*4+M3_XR]
    lea eax, [r11+r11*2]
    add eax, r10d
    add eax, r8d
    movss [r15+rax*4+M3_TMP], xmm0
    inc r11d
    cmp r11d, r9d
    jb .rof
    inc r10d
    cmp r10d, 3
    jb .row
    inc ebx
    jmp .ro
.rocopy:
    mov rax, [rsp+48]
    mov ecx, [rsp+60]
    movzx ecx, word [rax+rcx*2]
    lea ecx, [rcx+rcx*2]
.rc:
    mov eax, [r15+rcx*4+M3_TMP]
    mov [r15+rcx*4+M3_XR], eax
    inc ecx
    cmp ecx, 576
    jb .rc

.alias:
    ; alias reduction across subband edges: all of them for long blocks, the
    ; first only for mixed, none for short
    mov r9d, 32
    cmp dword [rsp+32], 1
    je .hyb
    jb .al
    mov r9d, 2
.al:
    mov ebx, 1
    lea r10, [acs]
    lea r11, [aca]
.asb:
    cmp ebx, r9d
    jae .hyb
    imul eax, ebx, 18
    lea rax, [r15+rax*4+M3_XR]
    xor ecx, ecx
.ai:
    mov edx, ecx
    not rdx                     ; -1 - i
    movss xmm0, [rax+rdx*4]     ; lower subband's top
    movss xmm1, [rax+rcx*4]     ; upper subband's bottom
    movss xmm2, [r10+rcx*4]
    movss xmm3, [r11+rcx*4]
    movaps xmm4, xmm0
    mulss xmm4, xmm2
    movaps xmm5, xmm1
    mulss xmm5, xmm3
    subss xmm4, xmm5
    movss [rax+rdx*4], xmm4
    mulss xmm1, xmm2
    mulss xmm0, xmm3
    addss xmm1, xmm0
    movss [rax+rcx*4], xmm1
    inc ecx
    cmp ecx, 8
    jb .ai
    inc ebx
    jmp .asb

.hyb:
    mov edx, [rsp+32]
    call hybrid
    call synth
    add rsp, 72
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; r15 = state, r14 = the granule's side info, edx = kind. XR through the imdct,
; overlap-add and frequency inversion into OUT
hybrid:
    push rbx
    push rsi
    push rdi
    push r12
    mov r12d, edx
    xor ebx, ebx
.sb:
    ; this subband's block type
    xor eax, eax
    cmp dword [r14+GR_WSF], 0
    je .bt
    mov eax, [r14+GR_BTYPE]
    cmp r12d, 2
    jne .bt
    cmp ebx, 2
    jae .bt
    xor eax, eax                ; mixed: the two lowest are long
.bt:
    imul esi, ebx, 18 * 4
    lea rsi, [r15+rsi+M3_XR]
    cmp eax, 2
    je .short
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    lea rdi, [cosl]
    xor ecx, ecx
.lk:
    vbroadcastss ymm5, [rsi+rcx*4]
    vfmadd231ps ymm0, ymm5, [rdi]
    vfmadd231ps ymm1, ymm5, [rdi+32]
    vfmadd231ps ymm2, ymm5, [rdi+64]
    vfmadd231ps ymm3, ymm5, [rdi+96]
    vfmadd231ps ymm4, ymm5, [rdi+128]
    add rdi, 160
    inc ecx
    cmp ecx, 18
    jb .lk
    imul eax, eax, 160
    lea rdi, [win]
    add rdi, rax
    vmulps ymm0, ymm0, [rdi]
    vmulps ymm1, ymm1, [rdi+32]
    vmulps ymm2, ymm2, [rdi+64]
    vmulps ymm3, ymm3, [rdi+96]
    vmulps ymm4, ymm4, [rdi+128]
    vmovups [r15+M3_Z], ymm0
    vmovups [r15+M3_Z+32], ymm1
    vmovups [r15+M3_Z+64], ymm2
    vmovups [r15+M3_Z+96], ymm3
    vmovups [r15+M3_Z+128], ymm4
    jmp .ovl
.short:
    ; three 12-point imdcts, windowed, overlapped at 6, 12 and 18
    vxorps ymm0, ymm0, ymm0
    vmovups [r15+M3_Z], ymm0
    vmovups [r15+M3_Z+32], ymm0
    vmovups [r15+M3_Z+64], ymm0
    vmovups [r15+M3_Z+96], ymm0
    vmovups [r15+M3_Z+128], ymm0
    xor edx, edx                ; window
.sw:
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    lea rdi, [coss]
    lea rcx, [rsi+rdx*4]
    xor eax, eax
.sk:
    vbroadcastss ymm5, [rcx]
    vfmadd231ps ymm0, ymm5, [rdi]
    vfmadd231ps ymm1, ymm5, [rdi+32]
    add rdi, 64
    add rcx, 12
    inc eax
    cmp eax, 6
    jb .sk
    vmovups [r15+M3_ZS], ymm0
    vmovups [r15+M3_ZS+32], ymm1
    imul eax, edx, 6
    lea rdi, [r15+rax*4+M3_Z+24]
    xor ecx, ecx
.sa:
    movss xmm0, [r15+rcx*4+M3_ZS]
    addss xmm0, [rdi+rcx*4]
    movss [rdi+rcx*4], xmm0
    inc ecx
    cmp ecx, 12
    jb .sa
    inc edx
    cmp edx, 3
    jb .sw
.ovl:
    imul edi, ebx, 18 * 4
    lea rdi, [r15+rdi+M3_OVL]
    xor ecx, ecx
.o:
    movss xmm0, [r15+rcx*4+M3_Z]
    addss xmm0, [rdi+rcx*4]
    mov eax, [r15+rcx*4+M3_Z+72]
    mov [rdi+rcx*4], eax
    mov eax, ebx
    and eax, ecx
    test eax, 1
    jz .noinv
    xorps xmm0, [c_sign]        ; odd subbands, odd samples
.noinv:
    mov eax, ecx
    shl eax, 5
    add eax, ebx
    movss [r15+rax*4+M3_OUT], xmm0
    inc ecx
    cmp ecx, 18
    jb .o
    inc ebx
    cmp ebx, 32
    jb .sb
    vzeroupper
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; r15 = state, r12 = out. OUT's 18 time slots through the polyphase filterbank:
; 576 samples
synth:
    push rbx
    push rsi
    push rdi
    xor ebx, ebx
.t:
    mov rdi, [r15+M3_VOFF]
    sub edi, 64
    and edi, 1023
    mov [r15+M3_VOFF], rdi
    ; the 64 new V values: matrixing of the 32 subband samples
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    mov esi, ebx
    shl esi, 7
    lea rsi, [r15+rsi+M3_OUT]
    lea rcx, [nt]
    xor edx, edx
.k:
    vbroadcastss ymm8, [rsi+rdx*4]
    vfmadd231ps ymm0, ymm8, [rcx]
    vfmadd231ps ymm1, ymm8, [rcx+32]
    vfmadd231ps ymm2, ymm8, [rcx+64]
    vfmadd231ps ymm3, ymm8, [rcx+96]
    vfmadd231ps ymm4, ymm8, [rcx+128]
    vfmadd231ps ymm5, ymm8, [rcx+160]
    vfmadd231ps ymm6, ymm8, [rcx+192]
    vfmadd231ps ymm7, ymm8, [rcx+224]
    add rcx, 256
    inc edx
    cmp edx, 32
    jb .k
    lea rax, [r15+rdi*4+M3_V]
    vmovups [rax], ymm0
    vmovups [rax+32], ymm1
    vmovups [rax+64], ymm2
    vmovups [rax+96], ymm3
    vmovups [rax+128], ymm4
    vmovups [rax+160], ymm5
    vmovups [rax+192], ymm6
    vmovups [rax+224], ymm7
    ; 32 samples: V's 16 windowed slices summed
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    lea rcx, [synth_d]
    xor edx, edx
.m:
    mov eax, edx
    shl eax, 7
    add eax, edi
    and eax, 1023
    lea r8, [r15+rax*4+M3_V]
    vmovups ymm4, [r8]
    vfmadd231ps ymm0, ymm4, [rcx]
    vmovups ymm4, [r8+32]
    vfmadd231ps ymm1, ymm4, [rcx+32]
    vmovups ymm4, [r8+64]
    vfmadd231ps ymm2, ymm4, [rcx+64]
    vmovups ymm4, [r8+96]
    vfmadd231ps ymm3, ymm4, [rcx+96]
    mov eax, edx
    shl eax, 7
    add eax, edi
    add eax, 96
    and eax, 1023
    lea r8, [r15+rax*4+M3_V]
    vmovups ymm4, [r8]
    vfmadd231ps ymm0, ymm4, [rcx+128]
    vmovups ymm4, [r8+32]
    vfmadd231ps ymm1, ymm4, [rcx+160]
    vmovups ymm4, [r8+64]
    vfmadd231ps ymm2, ymm4, [rcx+192]
    vmovups ymm4, [r8+96]
    vfmadd231ps ymm3, ymm4, [rcx+224]
    add rcx, 256
    inc edx
    cmp edx, 8
    jb .m
    mov eax, ebx
    shl eax, 7
    vmovups [r12+rax], ymm0
    vmovups [r12+rax+32], ymm1
    vmovups [r12+rax+64], ymm2
    vmovups [r12+rax+96], ymm3
    inc ebx
    cmp ebx, 18
    jb .t
    vzeroupper
    pop rdi
    pop rsi
    pop rbx
    ret

; r15 = state, rcx = a frame, edx = its bytes, r12 = where its 1152 samples go,
; r13 = rate index
frame:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rcx
    mov edi, edx
    ; side info, after the header and its crc
    lea rax, [rsi+4]
    test byte [rsi+1], 1
    jnz .nocrc
    add rax, 2
.nocrc:
    mov [r15+M3_BUF], rax
    mov qword [r15+M3_BP], 0
    gb 9
    mov [r15+M3_MDB], eax
    gb 5
    gb 4
    mov [r15+M3_SCFSI], eax
    lea rbx, [r15+M3_GR]
    xor r9d, r9d
.gi:
    gb 12
    mov [rbx+GR_P23], eax
    gb 9
    mov [rbx+GR_BIGV], eax
    gb 8
    mov [rbx+GR_GAIN], eax
    gb 4
    mov [rbx+GR_SCOMP], eax
    gb 1
    mov [rbx+GR_WSF], eax
    test eax, eax
    jz .nows
    gb 2
    mov [rbx+GR_BTYPE], eax
    gb 1
    mov [rbx+GR_MIXED], eax
    gb 5
    mov [rbx+GR_TSEL], eax
    gb 5
    mov [rbx+GR_TSEL+4], eax
    mov dword [rbx+GR_TSEL+8], 0
    gb 3
    mov [rbx+GR_SBG], eax
    gb 3
    mov [rbx+GR_SBG+4], eax
    gb 3
    mov [rbx+GR_SBG+8], eax
    jmp .gtail
.nows:
    mov dword [rbx+GR_BTYPE], 0
    mov dword [rbx+GR_MIXED], 0
    gb 5
    mov [rbx+GR_TSEL], eax
    gb 5
    mov [rbx+GR_TSEL+4], eax
    gb 5
    mov [rbx+GR_TSEL+8], eax
    gb 4
    mov [rbx+GR_R0], eax
    gb 3
    mov [rbx+GR_R1], eax
    mov dword [rbx+GR_SBG], 0
    mov dword [rbx+GR_SBG+4], 0
    mov dword [rbx+GR_SBG+8], 0
.gtail:
    gb 1
    mov [rbx+GR_PRE], eax
    gb 1
    mov [rbx+GR_SSCALE], eax
    gb 1
    mov [rbx+GR_C1SEL], eax
    add rbx, GR_SIZE
    inc r9d
    cmp r9d, 2
    jb .gi

    ; this frame's main data onto the reservoir
    mov rax, [r15+M3_BUF]
    add rax, 17
    lea rcx, [rsi+rdi]
    sub rcx, rax                ; its bytes
    jns .md
    xor ecx, ecx
.md:
    mov rdx, [r15+M3_NRES]
    lea r8, [rdx+rcx]
    cmp r8, RESCAP
    jbe .room
    ; full: keep the last 511 bytes (as far back as main_data_begin reaches)
    mov r8, rdx
    cmp r8, 511
    jbe .keep
    mov r8d, 511
.keep:
    lea r9, [r15+rdx+M3_RES]
    sub r9, r8
    xor r10d, r10d
.mv:
    cmp r10, r8
    jae .mvd
    mov r11b, [r9+r10]
    mov [r15+r10+M3_RES], r11b
    inc r10
    jmp .mv
.mvd:
    mov rdx, r8
.room:
    mov r8d, [r15+M3_MDB]
    mov r9, rdx
    sub r9, r8                  ; where this frame's granules start
    lea r10, [r15+rdx+M3_RES]
    xor r11d, r11d
.cp:
    cmp r11, rcx
    jae .cpd
    mov r8b, [rax+r11]
    mov [r10+r11], r8b
    inc r11
    jmp .cp
.cpd:
    add rdx, rcx
    mov [r15+M3_NRES], rdx
    mov qword [r15+rdx+M3_RES], 0   ; the bit reader looks 8 bytes ahead
    test r9, r9
    js .missing
    lea rax, [r15+M3_RES]
    mov [r15+M3_BUF], rax
    shl r9, 3
    mov [r15+M3_BP], r9
    xor ebx, ebx
    call granule
    add r12, 576 * 4
    mov ebx, 1
    call granule
    jmp .out
.missing:
    ; it needs bytes from before the start: silence
    xor eax, eax
.z:
    mov dword [r12+rax*4], 0
    inc eax
    cmp eax, 1152
    jb .z
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = the file's bytes, rdx = size, r8 = clip to fill (f32 mono at the file's rate,
; CL_DATA mem_alloc'd), r9 = a decoder state (M3_SIZE bytes, any contents). mpeg-1
; mono only (eax = 0 for the rest). eax = 1 if decoded. mp3_init first
global mp3_decode
mp3_decode:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 112
    mov rsi, rcx
    mov rdi, rdx
    mov rbx, r8
    mov r15, r9
    lea r8, [rsp+32]
    call mp3_info
    test eax, eax
    jz .fail
    cmp dword [rsp+32+MI_VER], 1
    jne .fail
    cmp dword [rsp+32+MI_MODES], 8
    jne .fail
    cmp dword [rsp+32+MI_MIXED], 0
    jne .fail
    mov eax, [rsp+32+MI_RATE]
    xor r13d, r13d
    cmp eax, 44100
    je .ri
    inc r13d
    cmp eax, 48000
    je .ri
    inc r13d
.ri:
    ; a clean state
    xor eax, eax
.cl:
    mov qword [r15+rax], 0
    add eax, 8
    cmp eax, M3_RES + 64
    jb .cl
    mov rcx, [rsp+32+MI_SAMPLES]
    lea rcx, [rcx*4+4608]
    call mem_alloc
    mov [rbx+CL_DATA], rax
    mov r12, rax
    ; the frames, as mp3_info found them
    xor r14d, r14d
    cmp rdi, 10
    jb .f
    mov eax, [rsi]
    and eax, 0xffffff
    cmp eax, 'ID3'
    jne .f
    xor eax, eax
    xor ecx, ecx
.syn:
    shl eax, 7
    movzx edx, byte [rsi+rcx+6]
    and edx, 0x7f
    or eax, edx
    inc ecx
    cmp ecx, 4
    jb .syn
    lea r14, [rax+10]
    test byte [rsi+5], 0x10
    jz .f
    add r14, 10
.f:
    lea rax, [r14+4]
    cmp rax, rdi
    ja .done
    mov ecx, [rsi+r14]
    bswap ecx
    call mp3_head
    test eax, eax
    jz .lost
    lea rcx, [r14+rax]
    cmp rcx, rdi
    ja .done
    mov [rsp+104], rax
    lea rcx, [rsi+r14]
    mov edx, eax
    call frame
    add r14, [rsp+104]
    add r12, 1152 * 4
    jmp .f
.lost:
    inc r14
    jmp .f
.done:
    mov rax, r12
    sub rax, [rbx+CL_DATA]
    shr rax, 2
    mov [rbx+CL_FRAMES], rax
    mov eax, [rsp+32+MI_RATE]
    mov [rbx+CL_RATE], eax
    mov dword [rbx+CL_CH], 1
    mov dword [rbx+CL_BITS], 16
    mov dword [rbx+CL_FLOAT], 0
    mov eax, 1
    jmp .out
.fail:
    xor eax, eax
.out:
    add rsp, 112
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
