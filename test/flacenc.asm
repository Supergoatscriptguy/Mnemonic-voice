; stage 3 test: the flac encoder. made-up signals and every librispeech test stream go
; through flac_encode and back through flac_decode bit for bit (the decoder checks the
; md5 and every crc). plus a stream built field by field, to reach what no encoder we
; have writes: rice2, escaped partitions (16 bit and 0 bit), a long unary run
; uses: audio\flac audio\flacenc data\parquet data\snappy data\zstd
default rel
bits 64
%include "lib.inc"
%include "test/check.inc"
%include "data/parquet.inc"
%include "audio/audio.inc"

NMAX    equ 70000

section .rdata
f_tc    db "datasets\librispeech\test.clean.parquet", 0
f_to    db "datasets\librispeech\test.other.parquet", 0
c_audio db "audio.bytes", 0
align 2
; what the hand-built stream holds
hand_a  dw 1, -1, 32767, -32768           ; escaped, 16 bits
hand_b  dw 100, -200, 30000, -30000       ; rice2, k = 20
hand_z  dw 0, 0, 0, 0                     ; escaped, 0 bits
hand_c  dw 0, 100, -1, 1                  ; k = 0: 100 is 200 zeros and a 1
align 8
c_m     dq 1e-6
c_pct   dq 100.0

section .bss
alignb 8
pcm     resw NMAX
fe      resb FE_SIZE
fd      resb FD_SIZE
pq      resb PQ_SIZEOF
ar      resb ARENA_SIZE
col     resb COL_SIZE
ra      resb RNG_SIZE
hand    resb 1024
handlen resq 1
orig    resq 1                  ; bytes: the original flacs, ours, raw pcm
ours    resq 1
raw     resq 1
nstr    resq 1
nbad    resq 1
ticks   resq 1
nsamp   resq 1

section .text

; rcx = int16 samples, rdx = how many. encodes them, decodes that, and compares.
; eax = 1 if it all came back exactly (md5 and crcs too)
roundtrip:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    mov rsi, rcx
    mov rbx, rdx
    call time_now
    mov rdi, rax
    mov rcx, rsi
    mov rdx, rbx
    mov r8d, 16000
    lea r9, [fe]
    call flac_encode
    call time_now
    sub rax, rdi
    add [ticks], rax
    add [nsamp], rbx
    mov rax, [fe+FE_LEN]
    add [ours], rax
    mov rcx, [fe+FE_DATA]
    mov rdx, [fe+FE_LEN]
    lea r8, [fd]
    call flac_decode
    xor edi, edi
    cmp eax, 1
    jne .free
    cmp dword [fd+FD_MD5], 1
    jne .free
    cmp [fd+FD_N], rbx
    jne .free
    push rsi
    mov rcx, rbx
    mov rdi, [fd+FD_PCM]
    repe cmpsw
    pop rsi
    mov edi, 0
    jne .free
    mov edi, 1
.free:
    mov rcx, [fd+FD_PCM]
    test rcx, rcx
    jz .nopcm
    call mem_free
.nopcm:
    mov rcx, [fe+FE_DATA]
    call mem_free
    mov eax, edi
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; edx = n. a 440 Hz sine at about half scale into pcm, by a recurrence
sine:
    push rbx
    lea r8, [pcm]
    xor ecx, ecx
    xor eax, eax                ; s[i-1]
    mov r9d, 3000               ; s[i]
.s:
    cmp ecx, edx
    jae .done
    mov [r8+rcx*2], r9w
    ; s[i+1] = 2 cos(w) s[i] - s[i-1], cos(2 pi 440/16000) ~ 0.98515, in 14 bit fixed point
    imul ebx, r9d, 16141
    sar ebx, 13
    sub ebx, eax
    mov eax, r9d
    mov r9d, ebx
    inc ecx
    jmp .s
.done:
    pop rbx
    ret

%macro trip 2+
    lea rcx, [pcm]
    mov edx, %1
    call roundtrip
    cmp eax, 1
    check e, %2
%endmacro

; the stream built by hand, into hand / handlen
build:
    push rdi
    push r14
    push r15
    push rbx
    sub rsp, 40
    lea rdi, [hand]
    xor r14d, r14d
    xor r15d, r15d
%macro p 2
    mov ecx, %1
    mov edx, %2
    call flac_put
%endmacro
    p 32, 0x664c6143            ; fLaC
    p 32, 0x80000022            ; the last metadata block: STREAMINFO, 34 bytes
    p 16, 16                    ; block sizes
    p 16, 16
    p 24, 0                     ; frame sizes unknown
    p 24, 0
    p 20, 16000
    p 3, 0                      ; mono
    p 5, 15                     ; 16 bits
    p 4, 0                      ; 36 bits of sample count
    p 32, 16
    p 32, 0                     ; no md5
    p 32, 0
    p 32, 0
    p 32, 0
    mov [rsp+32], rdi           ; the frame starts here
    p 16, 0xfff8
    p 4, 6                      ; block size: 8 bits of size - 1 follow
    p 4, 5                      ; 16 kHz
    p 4, 0
    p 3, 4
    p 1, 0
    p 8, 0                      ; frame 0
    p 8, 15
    mov rcx, [rsp+32]
    mov rdx, rdi
    call flac_crc8
    p 8, eax
    p 8, 0x10                   ; fixed, order 0
    p 2, 1                      ; rice2
    p 4, 2                      ; 4 partitions of 4
    ; escaped, 16 bit raw samples
    p 5, 31
    p 5, 16
    xor ebx, ebx
.a:
    lea rax, [hand_a]
    movzx edx, word [rax+rbx*2]
    mov ecx, 16
    call flac_put
    inc ebx
    cmp ebx, 4
    jb .a
    ; rice2, k = 20: no zeros, the 1, then 20 bits of the zigzag value
    p 5, 20
    xor ebx, ebx
.b:
    lea rax, [hand_b]
    movsx eax, word [rax+rbx*2]
    lea edx, [rax+rax]
    sar eax, 31
    xor edx, eax
    bts edx, 20
    mov ecx, 21
    call flac_put
    inc ebx
    cmp ebx, 4
    jb .b
    ; escaped with 0 bits: four zeros, nothing stored
    p 5, 31
    p 5, 0
    ; k = 0: zigzag 0 1 200 2 -> "1", 200 zeros and "1", "01", "001"
    p 5, 0
    p 1, 1
    mov ebx, 6
.z:
    p 32, 0
    dec ebx
    jnz .z
    p 8, 0                      ; 6 x 32 + 8 = 200
    p 1, 1
    p 2, 1
    p 3, 1
    test r15d, r15d
    jz .padded
    mov ecx, 8
    sub ecx, r15d
    xor edx, edx
    call flac_put
.padded:
    mov rcx, [rsp+32]
    mov rdx, rdi
    call flac_crc16
    p 16, eax
    lea rax, [hand]
    sub rdi, rax
    mov [handlen], rdi
    add rsp, 40
    pop rbx
    pop r15
    pop r14
    pop rdi
    ret

; rcx = parquet. every audio.bytes value: decode, our encode, decode again
parquet:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 48
    mov rdx, rcx
    lea rcx, [pq]
    call pq_open
    test eax, eax
    jz .out
    lea rcx, [pq]
    lea rdx, [c_audio]
    call pq_leaf
    mov r12, rax
    xor ebx, ebx
.rg:
    cmp rbx, [pq+PQ_NRG]
    jae .closed
    lea rcx, [pq]
    lea rdx, [col]
    mov r8, rbx
    mov r9, r12
    call pq_chunk
    xor edi, edi
.v:
    cmp rdi, [col+COL_NV]
    jae .rgdone
    mov rax, [col+COL_VALS]
    mov rdx, rdi
    shl rdx, 4
    mov rcx, [rax+rdx]
    mov rdx, [rax+rdx+8]
    add [orig], rdx
    lea r8, [fd]
    call flac_decode
    cmp eax, 1
    jne .bad
    ; keep the samples: roundtrip reuses fd
    mov r13, [fd+FD_N]
    mov rsi, [fd+FD_PCM]
    lea rax, [r13+r13]
    add [raw], rax
    mov rcx, rsi
    mov rdx, r13
    call roundtrip
    mov [rsp+32], eax
    mov rcx, rsi
    call mem_free
    cmp dword [rsp+32], 1
    je .good
.bad:
    inc qword [nbad]
.good:
    inc qword [nstr]
    inc rdi
    jmp .v
.rgdone:
    lea rcx, [ar]
    call arena_reset
    inc rbx
    jmp .rg
.closed:
    lea rcx, [pq]
    call pq_close
.out:
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

    say "round trips of made-up signals", 13, 10
    lea rdi, [pcm]
    xor eax, eax
    mov ecx, NMAX
    rep stosw
    trip 10000, "  silence (constant subframes)"
    lea rcx, [ra]
    mov edx, 5
    call rng_seed
    xor ebx, ebx
.noise:
    lea rcx, [ra]
    call rng_next
    lea rcx, [pcm]
    mov [rcx+rbx*2], ax
    inc ebx
    cmp ebx, 20000
    jb .noise
    trip 20000, "  full scale noise (verbatim)"
    mov edx, 16000
    call sine
    lea rcx, [pcm]
    xor eax, eax
.spike:
    mov word [rcx+rax*2], 30000
    add eax, 997
    cmp eax, 16000
    jb .spike
    trip 16000, "  a sine with spikes (long unary codes)"
    lea rcx, [pcm]
    xor eax, eax
.ramp:
    mov edx, eax
    sub edx, 32768
    mov [rcx+rax*2], dx
    inc eax
    cmp eax, 65536
    jb .ramp
    trip 65536, "  a ramp through every 16 bit value"
    mov edx, 9000
    call sine
    trip 1, "  1 sample"
    trip 2, "  2 samples"
    trip 5, "  5 samples (shorter than the orders want)"
    trip 4095, "  4095 samples"
    trip 4096, "  4096 samples (one whole block)"
    trip 4097, "  4097 samples (a block and one)"
    trip 8192, "  8192 samples"

    say "a stream built by hand", 13, 10
    call build
    lea rcx, [hand]
    mov rdx, [handlen]
    lea r8, [fd]
    call flac_decode
    cmp eax, 1
    check e, "  decodes"
    cmp qword [fd+FD_N], 16
    check e, "  16 samples"
    mov rsi, [fd+FD_PCM]
    lea rdi, [hand_a]
    mov ecx, 16
    repe cmpsw
    check e, "  escaped, rice2 k=20, escaped 0 bit, k=0 come out right"
    cmp qword [fd+FD_STATS+8*8], 1
    check e, "  the rice2 path ran"
    cmp qword [fd+FD_STATS+9*8], 2
    check e, "  the escape path ran twice"
    cmp qword [fd+FD_STATS+13*8], 1
    check ae, "  the long unary path ran"
    mov rcx, [fd+FD_PCM]
    call mem_free

    lea rcx, [f_tc]
    call file_exists
    test eax, eax
    jz .nols
    say "every librispeech test stream: decode, our encode, decode", 13, 10
    lea rcx, [ar]
    mov rdx, 1 << 34
    call arena_init
    lea rax, [ar]
    mov [col+COL_ARENA], rax
    mov ecx, ZSTD_CTX
    call mem_alloc
    mov [col+COL_ZSTD], rax
    mov rcx, rax
    call zstd_init
    xor eax, eax
    mov [ticks], rax
    mov [nsamp], rax
    mov [ours], rax
    lea rcx, [f_tc]
    call parquet
    lea rcx, [f_to]
    call parquet
    say "  "
    mov rcx, [nstr]
    call print_dec
    say " streams. pcm "
    mov rcx, [raw]
    call print_dec
    say " bytes, libFLAC "
    cvtsi2sd xmm0, qword [orig]
    cvtsi2sd xmm1, qword [raw]
    divsd xmm0, xmm1
    mulsd xmm0, [c_pct]
    mov edx, 1
    call print_fixed
    say "%, ours "
    cvtsi2sd xmm0, qword [ours]
    cvtsi2sd xmm1, qword [raw]
    divsd xmm0, xmm1
    mulsd xmm0, [c_pct]
    mov edx, 1
    call print_fixed
    say "%. encoded at "
    cvtsi2sd xmm1, qword [ticks]
    cvtsi2sd xmm0, qword [qpc_freq]
    divsd xmm1, xmm0
    cvtsi2sd xmm0, qword [nsamp]
    divsd xmm0, xmm1
    mulsd xmm0, [c_m]
    xor edx, edx
    call print_fixed
    say "M samples/s on one thread", 13, 10
    cmp qword [nbad], 0
    check e, "  every one comes back bit for bit"
    jmp t_done
.nols:
    say "(no librispeech test sets in datasets\, skipped)", 13, 10
    jmp t_done
