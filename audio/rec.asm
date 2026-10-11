; rec [secs=10] [out=recordings\rec.wav] [out16=recordings\rec16k.wav] [play=0] [loopback=1]
; records from the default mic (or with loopback=1, whatever the default output is
; playing) with a live level meter, ctrl+c stops early. out keeps the device's own
; rate and channels, out16 is mono 16 kHz like the speech model's input, and that
; one gets played back unless play=0
; uses: audio\wav audio\dev audio\resample
; libs: ole32.lib
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern ExitProcess

BARW equ 40

section .rdata
align 8
c_db    dq 8.685889638065035    ; 20 / ln 10
c_floor dq 1e-5                 ; -100 dB
c_m100  dq -100.0
c_60    dq 60.0
c_barw  dq 40.0
c_m12   dq -12.0
c_m3    dq -3.0
c_clip  dq -0.1
c_secs  dq 10.0
c_abs   dq 0x7fffffffffffffff
k_secs  db "secs", 0
k_out   db "out", 0
k_loop  db "loopback", 0
k_out16 db "out16", 0
d_out16 db "recordings\rec16k.wav", 0
k_play  db "play", 0
d_out   db "recordings\rec.wav", 0
p_dir   db "recordings", 0

section .bss
alignb 8
dev     resb DV_SIZE
clip    resb CL_SIZE
clip16  resb CL_SIZE
clipout resb CL_SIZE
line    resb 512

section .text

; xmm0 = linear level. xmm0 = dBFS, floored at -100
todb:
    sub rsp, 40
    comisd xmm0, [c_floor]
    jb .floor
    call math_log
    mulsd xmm0, [c_db]
    add rsp, 40
    ret
.floor:
    movsd xmm0, [c_m100]
    add rsp, 40
    ret

; xmm0 = peak, xmm1 = rms (linear), xmm2 = seconds. redraws the meter line
meter:
    push rsi
    push rdi
    sub rsp, 56
    movsd [rsp+32], xmm1
    movsd [rsp+40], xmm2
    call todb
    movsd [rsp+48], xmm0
    movsd xmm0, [rsp+32]
    call todb
    movsd [rsp+32], xmm0

    lea rdi, [line]
    emit 13, "  "
    mov rcx, rdi
    movsd xmm1, [rsp+40]
    mov r8d, 1
    call fmt_fixed
    mov rdi, rax
    emit " s  "
    ; green, yellow above -12, red above -3
    movsd xmm0, [rsp+48]
    comisd xmm0, [c_m3]
    ja .red
    comisd xmm0, [c_m12]
    ja .yellow
    emit 27, "[32m"
    jmp .bar
.yellow:
    emit 27, "[33m"
    jmp .bar
.red:
    emit 27, "[31m"
.bar:
    ; the bar is the peak on a -60..0 dB scale
    movsd xmm0, [rsp+48]
    addsd xmm0, [c_60]
    divsd xmm0, [c_60]
    mulsd xmm0, [c_barw]
    cvttsd2si ecx, xmm0
    xor eax, eax
    test ecx, ecx
    cmovs ecx, eax
    mov eax, BARW
    cmp ecx, eax
    cmova ecx, eax
    mov edx, BARW
    sub edx, ecx
.full:
    test ecx, ecx
    jz .light
    mov word [rdi], 0x96e2      ; U+2588 full block
    mov byte [rdi+2], 0x88
    add rdi, 3
    dec ecx
    jmp .full
.light:
    emit 27, "[90m"
.empty:
    test edx, edx
    jz .nums
    mov word [rdi], 0x96e2      ; U+2591 light shade
    mov byte [rdi+2], 0x91
    add rdi, 3
    dec edx
    jmp .empty
.nums:
    emit 27, "[0m  peak "
    mov rcx, rdi
    movsd xmm1, [rsp+48]
    mov r8d, 1
    call fmt_fixed
    mov rdi, rax
    emit "  rms "
    mov rcx, rdi
    movsd xmm1, [rsp+32]
    mov r8d, 1
    call fmt_fixed
    mov rdi, rax
    emit " dBFS"
    movsd xmm0, [rsp+48]
    comisd xmm0, [c_clip]
    jb .eol
    emit "  ", 27, "[31mCLIP", 27, "[0m"
.eol:
    emit 27, "[K"
    lea rcx, [line]
    mov rdx, rdi
    sub rdx, rcx
    call print
    add rsp, 56
    pop rdi
    pop rsi
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 80
    ; locals: peak +40, sum of squares +48, chunk start +56, out path +64, secs +72
    call lib_init
    call ctrlc_init
    call cfg_args
    lea rcx, [k_secs]
    movsd xmm1, [c_secs]
    call cfg_float
    movsd [rsp+72], xmm0
    lea rcx, [k_out]
    lea rdx, [d_out]
    call cfg_str
    mov [rsp+64], rax

    lea rcx, [k_loop]
    xor edx, edx
    call cfg_int
    mov ecx, 1
    test rax, rax
    jz .mic
    mov ecx, 2
.mic:
    lea rdx, [dev]
    mov r8d, 50
    call dev_open
    test eax, eax
    jnz .opened
    say "the mic doesn't mix in float32, not handled yet", 13, 10
    mov ecx, 1
    call ExitProcess
.opened:
    lea rcx, [dev]
    call dev_show

    ; room for secs of audio plus a buffer's worth, since packets come whole
    cvtsi2sd xmm0, dword [dev+DV_RATE]
    mulsd xmm0, [rsp+72]
    cvttsd2si r13, xmm0         ; frames wanted
    mov r14d, [dev+DV_CH]
    mov eax, [dev+DV_BUF]
    lea r15, [r13+rax]          ; capacity
    mov rcx, r15
    imul rcx, r14
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov rbx, rax
    xor r12d, r12d              ; frames so far
    mov eax, [dev+DV_RATE]
    xor edx, edx
    mov ecx, 20
    div ecx
    mov esi, eax                ; meter window, 50 ms
    xor edi, edi                ; frames in it so far
    xorpd xmm0, xmm0
    movsd [rsp+40], xmm0
    movsd [rsp+48], xmm0

    say "recording, ctrl+c to stop", 13, 10
    lea rcx, [dev]
    call dev_start
.loop:
    cmp dword [stop_flag], 0
    jne .stop
    cmp r12, r13
    jae .stop
    lea rcx, [dev]
    call dev_wait
    test eax, eax
    jz .ready
    cmp dword [dev+DV_DIR], 2   ; loopback gets no events while nothing plays: done
    je .stop
    jmp .dead
.ready:
    mov rdx, r12
    imul rdx, r14
    lea rdx, [rbx+rdx*4]
    mov [rsp+56], rdx
    mov r8, r15
    sub r8, r12
    lea rcx, [dev]
    call dev_read
    add r12, rax
    add edi, eax
    mov rcx, rax
    imul rcx, r14
    mov rdx, [rsp+56]
    movsd xmm2, [rsp+40]
    movsd xmm3, [rsp+48]
    movsd xmm4, [c_abs]
    test rcx, rcx
    jz .chunked
.sample:
    cvtss2sd xmm0, [rdx]
    andpd xmm0, xmm4
    maxsd xmm2, xmm0
    mulsd xmm0, xmm0
    addsd xmm3, xmm0
    add rdx, 4
    dec rcx
    jnz .sample
.chunked:
    movsd [rsp+40], xmm2
    movsd [rsp+48], xmm3
    cmp edi, esi
    jb .loop
    mov eax, edi
    imul eax, r14d
    cvtsi2sd xmm0, rax
    divsd xmm3, xmm0
    sqrtsd xmm1, xmm3
    movsd xmm0, xmm2
    cvtsi2sd xmm2, r12
    cvtsi2sd xmm5, dword [dev+DV_RATE]
    divsd xmm2, xmm5
    call meter
    xor edi, edi
    xorpd xmm0, xmm0
    movsd [rsp+40], xmm0
    movsd [rsp+48], xmm0
    jmp .loop

.stop:
    lea rcx, [dev]
    call dev_stop
    say 13, 10, "recorded "
    cvtsi2sd xmm0, r12
    cvtsi2sd xmm1, dword [dev+DV_RATE]
    divsd xmm0, xmm1
    mov edx, 2
    call print_fixed
    say " s, "
    mov ecx, [dev+DV_GLITCH]
    call print_dec
    say " glitches, "
    mov ecx, [dev+DV_PACKETS]
    call print_dec
    say " packets, "
    mov ecx, [dev+DV_SILENT]
    call print_dec
    say " flagged silent", 13, 10

    mov [clip+CL_DATA], rbx
    mov [clip+CL_FRAMES], r12
    mov eax, [dev+DV_RATE]
    mov [clip+CL_RATE], eax
    mov [clip+CL_CH], r14d
    lea rcx, [p_dir]
    call make_dir
    mov rcx, [rsp+64]
    lea rdx, [clip]
    mov r8d, 32
    call wav_write
    test eax, eax
    jnz .written
    say "couldn't write the wav", 13, 10
    jmp .close
.written:
    say "wrote "
    mov rcx, [rsp+64]
    call print_z
    say 13, 10
.close:
    lea rcx, [dev]
    call dev_close

    ; what the speech model will hear: mono, 16 kHz
    lea rcx, [clip]
    call clip_mono
    lea rcx, [clip]
    mov edx, 16000
    lea r8, [clip16]
    call clip_resample
    lea rcx, [k_out16]
    lea rdx, [d_out16]
    call cfg_str
    mov rbx, rax
    mov rcx, rax
    lea rdx, [clip16]
    mov r8d, 16
    call wav_write
    say "wrote "
    mov rcx, rbx
    call print_z
    say " (16 kHz mono, what the model hears)", 13, 10

    lea rcx, [k_play]
    mov edx, 1
    call cfg_int
    test rax, rax
    jz .bye
    ; play that back, so you hear exactly what the model gets
    lea rdi, [dev]
    mov ecx, DV_SIZE
    xor eax, eax
    rep stosb
    xor ecx, ecx
    lea rdx, [dev]
    mov r8d, 50
    call dev_open
    test eax, eax
    jz .bye
    lea rcx, [clip16]
    mov edx, [dev+DV_RATE]
    lea r8, [clipout]
    call clip_resample
    say "playing it back on "
    lea rcx, [dev]
    call dev_show
    lea rcx, [dev]
    lea rdx, [clipout]
    call dev_play
    lea rcx, [dev]
    call dev_close
.bye:
    call con_restore
    xor ecx, ecx
    call ExitProcess
.dead:
    say 13, 10, "the mic stopped delivering audio", 13, 10
    call con_restore
    mov ecx, 1
    call ExitProcess
