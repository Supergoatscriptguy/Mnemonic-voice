; stage 3 test: our mp3 decoder against the one windows ships (the fraunhofer acm
; codec, l3codeca.acm). every 40th clip of the common voice spontaneous set goes
; through both; the first pair finds how far apart their outputs sit, then each
; clip's snr of ours against its 16-bit output
; uses: audio\mp3 data\tar
; libs: msacm32.lib
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"
%include "data/tar.inc"
%include "test/check.inc"

extern acmStreamOpen, acmStreamSize, acmStreamPrepareHeader, acmStreamConvert
extern acmStreamUnprepareHeader, acmStreamClose

EVERY   equ 40
MAXLAG  equ 2304
WIN     equ 4096
HDRSZ   equ 128                 ; ACMSTREAMHEADER on x64

section .rdata
align 8
c_10log dq 4.342944819032518    ; 10 / ln 10
c_i16   dq 3.0517578125e-05
c_big   dq 1e30
c_40    dq 40.0
c_m1    dq -1.0
c_p1    dq 0.999969482421875    ; 32767 / 32768
c_m80   dq -80.0
c_tiny  dq 1e-30
c_m300  dq -300.0
f_tar   db "datasets\cv\cv-spontaneous-5.0-en.tar", 0
n_snr   db "worst clip's error against windows' decoder (dBFS), under -80", 0
n_all   db "every sampled clip decoded by both", 0

section .bss
alignb 8
ta      resb TA_SIZEOF
st      resq 1
clip    resb CL_SIZE
mi      resb MI_SIZE
has     resq 1
hdr     resb HDRSZ
srcfmt  resb 32
dstfmt  resb 24
lag     resq 1
haslag  resq 1
worst   resq 1
sum     resq 1
nclips  resq 1
nsnr    resq 1
nfail   resq 1
secs    resq 1
tdec    resq 1
pcm     resq 1                  ; windows' output
npcm    resq 1
e_err   resq 1                  ; the last compare's sums
e_sig   resq 1
e_n     resq 1
name    resq 1

section .text

; rcx = an mp3, rdx = its size (mi filled). windows' decoding of it into pcm/npcm,
; eax = 1 if that went
acm:
    push rbx
    push rsi
    push rdi
    sub rsp, 64
    mov rsi, rcx
    mov rdi, rdx
    ; MPEGLAYER3WAVEFORMAT in, 16-bit pcm out
    lea rbx, [srcfmt]
    mov word [rbx], 0x55
    mov word [rbx+2], 1
    mov eax, [mi+MI_RATE]
    mov [rbx+4], eax
    imul eax, [mi+MI_MINBR], 125
    mov [rbx+8], eax
    mov word [rbx+12], 1
    mov word [rbx+14], 0
    mov word [rbx+16], 12
    mov word [rbx+18], 1        ; MPEGLAYER3_ID_MPEG
    mov dword [rbx+20], 0       ; iso padding
    imul eax, [mi+MI_MINBR], 144000
    xor edx, edx
    div dword [mi+MI_RATE]
    mov [rbx+24], ax            ; a frame's bytes
    mov word [rbx+26], 1
    mov word [rbx+28], 0
    lea rbx, [dstfmt]
    mov word [rbx], 1
    mov word [rbx+2], 1
    mov eax, [mi+MI_RATE]
    mov [rbx+4], eax
    add eax, eax
    mov [rbx+8], eax
    mov word [rbx+12], 2
    mov word [rbx+14], 16
    mov word [rbx+16], 0
    lea rcx, [has]
    xor edx, edx
    lea r8, [srcfmt]
    lea r9, [dstfmt]
    mov qword [rsp+32], 0
    mov qword [rsp+40], 0
    mov qword [rsp+48], 0
    mov qword [rsp+56], 4       ; ACM_STREAMOPENF_NONREALTIME
    call acmStreamOpen
    test eax, eax
    jnz .err
    mov rcx, [has]
    mov edx, edi
    lea r8, [rsp+60]
    mov dword [r8], 0
    xor r9d, r9d
    call acmStreamSize
    mov ecx, [rsp+60]
    add ecx, 65536
    call mem_alloc
    mov [pcm], rax
    lea rbx, [hdr]
    xor eax, eax
.z:
    mov qword [rbx+rax], 0
    add eax, 8
    cmp eax, HDRSZ
    jb .z
    mov dword [rbx], HDRSZ
    mov [rbx+16], rsi
    mov [rbx+24], edi
    mov rax, [pcm]
    mov [rbx+40], rax
    mov eax, [rsp+60]
    add eax, 65536
    mov [rbx+48], eax
    mov rcx, [has]
    mov rdx, rbx
    xor r8d, r8d
    call acmStreamPrepareHeader
    test eax, eax
    jnz .err
    mov rcx, [has]
    lea rdx, [hdr]
    mov r8d, 0x34               ; block align, start, end
    call acmStreamConvert
    test eax, eax
    jnz .err
    mov eax, [hdr+52]
    shr eax, 1
    mov [npcm], rax
    mov rcx, [has]
    lea rdx, [hdr]
    xor r8d, r8d
    call acmStreamUnprepareHeader
    mov rcx, [has]
    xor edx, edx
    call acmStreamClose
    mov eax, 1
    jmp .out
.err:
    mov ebx, eax
    say "  acm said "
    mov ecx, ebx
    call print_dec
    say 13, 10
    xor eax, eax
.out:
    add rsp, 64
    pop rdi
    pop rsi
    pop rbx
    ret

; ours (clip) against windows' (pcm) with ecx = lag: windows' sample i + lag is
; our sample i. xmm0 = error energy / signal energy over WIN samples from 8192,
; or over all of the overlap if rdx = 1. xmm1 = signal energy
compare:
    push rbx
    push rsi
    mov r8, [clip+CL_DATA]
    mov r9, [pcm]
    movsxd r10, ecx
    xorpd xmm0, xmm0            ; error
    xorpd xmm1, xmm1            ; signal
    movsd xmm4, [c_i16]
    ; i from max(0, -lag) to min(ours, theirs - lag)
    xor esi, esi
    test r10, r10
    jns .lo
    mov rsi, r10
    neg rsi
.lo:
    mov rbx, [npcm]
    sub rbx, r10
    cmp rbx, [clip+CL_FRAMES]
    jbe .hi
    mov rbx, [clip+CL_FRAMES]
.hi:
    test edx, edx
    jnz .run0
    mov eax, 8192
    cmp rsi, rax
    cmovb rsi, rax
    lea rax, [rsi+WIN]
    cmp rax, rbx
    cmovb rbx, rax
.run0:
    mov r11, rsi
.run:
    cmp rsi, rbx
    jge .done
    lea rax, [rsi+r10]
    movsx eax, word [r9+rax*2]
    cvtsi2sd xmm2, eax
    mulsd xmm2, xmm4
    cvtss2sd xmm3, [r8+rsi*4]
    maxsd xmm3, [c_m1]          ; windows clips to 16 bits, so we do here too
    minsd xmm3, [c_p1]
    movapd xmm5, xmm2
    mulsd xmm5, xmm5
    addsd xmm1, xmm5
    subsd xmm3, xmm2
    mulsd xmm3, xmm3
    addsd xmm0, xmm3
    inc rsi
    jmp .run
.done:
    movsd [e_err], xmm0
    movsd [e_sig], xmm1
    sub rbx, r11
    mov [e_n], rbx
    xorpd xmm2, xmm2
    comisd xmm1, xmm2
    jbe .flat
    divsd xmm0, xmm1
    pop rsi
    pop rbx
    ret
.flat:
    movsd xmm0, [c_big]
    pop rsi
    pop rbx
    ret

; xmm0 = an energy summed over e_n samples: prints its rms in dBFS
per_db:
    sub rsp, 40
    cvtsi2sd xmm1, qword [e_n]
    divsd xmm0, xmm1
    xorpd xmm1, xmm1
    comisd xmm0, xmm1
    ja .log
    say "-inf"
    jmp .out
.log:
    call math_log
    mulsd xmm0, [c_10log]
    mov edx, 1
    call print_fixed
.out:
    add rsp, 40
    ret

; rcx = an mp3, rdx = its size
one:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov rsi, rcx
    mov rdi, rdx
    lea r8, [mi]
    call mp3_info
    call time_now
    mov rbx, rax
    mov rcx, rsi
    mov rdx, rdi
    lea r8, [clip]
    mov r9, [st]
    call mp3_decode
    mov r12d, eax
    mov rcx, rbx
    call time_since
    addsd xmm0, [tdec]
    movsd [tdec], xmm0
    test r12d, r12d
    jz .bad
    mov rcx, rsi
    mov rdx, rdi
    call acm
    test eax, eax
    jz .bad
    cvtsi2sd xmm0, qword [clip+CL_FRAMES]
    cvtsi2sd xmm1, dword [clip+CL_RATE]
    divsd xmm0, xmm1
    addsd xmm0, [secs]
    movsd [secs], xmm0
    cmp qword [haslag], 0
    jne .snr
    ; the first one: where the outputs line up
    mov ebx, -MAXLAG
    movsd xmm0, [c_big]
    movsd [rsp+32], xmm0
.lag:
    mov ecx, ebx
    xor edx, edx
    call compare
    comisd xmm0, [rsp+32]
    jae .lagn
    movsd [rsp+32], xmm0
    mov [lag], rbx
.lagn:
    inc ebx
    cmp ebx, MAXLAG
    jle .lag
    mov qword [haslag], 1
    say "  windows' output sits "
    mov rcx, [lag]
    call print_int
    say " samples from ours", 13, 10
.snr:
    mov rcx, [lag]
    mov edx, 1
    call compare
    ; snr, where there's a signal: -10 log10(err / signal)
    xorpd xmm1, xmm1
    comisd xmm1, [e_sig]
    jae .err
    maxsd xmm0, [c_tiny]
    call math_log
    mulsd xmm0, [c_10log]
    xorpd xmm1, xmm1
    subsd xmm1, xmm0
    addsd xmm1, [sum]
    movsd [sum], xmm1
    inc qword [nsnr]
.err:
    ; and the error's own level, in dBFS (16-bit rounding alone is about -101)
    movsd xmm0, [e_err]
    cvtsi2sd xmm1, qword [e_n]
    divsd xmm0, xmm1
    maxsd xmm0, [c_tiny]
    call math_log
    mulsd xmm0, [c_10log]
    movsd [rsp+32], xmm0
    comisd xmm0, [worst]
    jbe .nw
    movsd [worst], xmm0
.nw:
    movsd xmm0, [rsp+32]
    comisd xmm0, [c_m80]
    jbe .fine
    say "  "
    mov rcx, [name]
    call print_z
    say ": error "
    movsd xmm0, [rsp+32]
    mov edx, 1
    call print_fixed
    say " dBFS, ours "
    mov rcx, [clip+CL_FRAMES]
    call print_dec
    say " samples, windows' "
    mov rcx, [npcm]
    call print_dec
    say ", compared "
    mov rcx, [e_n]
    call print_dec
    say ", signal "
    movsd xmm0, [e_sig]
    call per_db
    say " dBFS", 13, 10
.fine:
    inc qword [nclips]
    jmp .free
.bad:
    inc qword [nfail]
.free:
    mov rcx, [clip+CL_DATA]
    test rcx, rcx
    jz .f2
    call mem_free
    mov qword [clip+CL_DATA], 0
.f2:
    mov rcx, [pcm]
    test rcx, rcx
    jz .out
    call mem_free
    mov qword [pcm], 0
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
    call mp3_init
    mov ecx, M3_SIZE
    call mem_alloc
    mov [st], rax
    movsd xmm0, [c_m300]
    movsd [worst], xmm0
    lea rcx, [f_tar]
    call file_map
    test rax, rax
    jnz .mapped
    say "  skipped: no datasets\cv\cv-spontaneous-5.0-en.tar", 13, 10
    jmp t_done
.mapped:
    lea rcx, [ta]
    mov r8, rdx
    mov rdx, rax
    call tar_open
    xor esi, esi                ; mp3s seen
.member:
    lea rcx, [ta]
    call tar_next
    test eax, eax
    jz .walked
    mov rcx, [ta+TA_NAME]
.end:
    cmp byte [rcx], 0
    je .ext
    inc rcx
    jmp .end
.ext:
    cmp dword [rcx-4], '.mp3'
    jne .member
    inc esi
    mov eax, esi
    xor edx, edx
    mov ecx, EVERY
    div ecx
    test edx, edx
    jnz .member
    mov rax, [ta+TA_NAME]
    mov [name], rax
    mov rcx, [ta+TA_DATA]
    mov rdx, [ta+TA_SIZE]
    call one
    jmp .member
.walked:
    say "  "
    mov rcx, [nclips]
    call print_dec
    say " clips, "
    movsd xmm0, [secs]
    mov edx, 1
    call print_fixed
    say " s of audio, mean snr "
    movsd xmm0, [sum]
    cvtsi2sd xmm1, qword [nsnr]
    divsd xmm0, xmm1
    mov edx, 1
    call print_fixed
    say " dB. ours decodes at "
    movsd xmm0, [secs]
    divsd xmm0, [tdec]
    xor edx, edx
    call print_fixed
    say "x real time on one thread", 13, 10
    xor ecx, ecx
    cmp qword [nfail], 0
    sete cl
    lea rdx, [n_all]
    call t_ok
    movsd xmm2, [worst]
    xor ecx, ecx
    comisd xmm2, [c_m80]
    setb cl
    lea rdx, [n_snr]
    call t_okf
    jmp t_done
