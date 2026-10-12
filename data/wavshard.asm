; wavshard src=N out=FILE.shard [match=TEXT] ARCHIVE.tar|.zip...
; every .wav in the archives whose path has TEXT in it -> one shard, mono, 16 kHz,
; the path as its id and no text. for the noise and room impulse response sets
; uses: data\tar data\zip data\inflate data\shardw audio\wav audio\resample audio\flac audio\flacenc
default rel
bits 64
%include "lib.inc"
%include "data/tar.inc"
%include "data/zip.inc"
%include "data/shard.inc"
%include "audio/audio.inc"

extern ExitProcess

section .rdata
k_src   db "src", 0
k_out   db "out", 0
k_match db "match", 0
usage   db "usage: wavshard src=N out=FILE.shard [match=TEXT] ARCHIVE.tar|.zip...", 13, 10, 0
e_open  db "can't read an archive", 0
e_out   db "can't create the shard", 0
align 8
c_rate  dq 16000.0
c_hour  dq 3600.0
c_scale dd 32768.0
c_lo    dd -32768.0
c_hi    dd 32767.0

section .bss
alignb 8
ta      resb TA_SIZEOF
zp      resb ZP_SIZEOF
sw      resb SW_SIZE
ua      resb UA_SIZE
clip    resb CL_SIZE
conv    resb CL_SIZE
match   resq 1
src     resq 1
nwav    resq 1
nbad    resq 1
samples resq 1
archives resq 64
narch   resq 1

section .text

; rcx = path (zero terminated). eax = 1 if it ends in .wav and has the match in it
wanted:
    mov rdx, rcx
.end:
    cmp byte [rdx], 0
    je .ended
    inc rdx
    jmp .end
.ended:
    lea rax, [rdx-4]
    cmp rax, rcx
    jb .no
    mov eax, [rdx-4]
    or eax, 0x20202000          ; lower case the letters
    cmp eax, '.wav'
    jne .no
    mov r8, [match]
    test r8, r8
    jz .yes
.at:
    cmp byte [rcx], 0
    je .no
    mov r9, rcx
    mov r10, r8
.cmp:
    mov al, [r10]
    test al, al
    jz .yes
    cmp al, [r9]
    jne .next
    inc r9
    inc r10
    jmp .cmp
.next:
    inc rcx
    jmp .at
.yes:
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; rcx = wav bytes, rdx = length, r8 = its name. into the shard
onewav:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, r8
    lea r8, [clip]
    call wav_parse
    test eax, eax
    jnz .parsed
    inc qword [nbad]
    jmp .out
.parsed:
    lea rcx, [clip]
    call clip_mono
    lea rsi, [clip]
    cmp dword [clip+CL_RATE], 16000
    je .int16
    lea rcx, [clip]
    mov edx, 16000
    lea r8, [conv]
    call clip_resample
    mov rcx, [clip+CL_DATA]
    call mem_free
    lea rsi, [conv]
.int16:
    mov rcx, [rsi+CL_FRAMES]
    mov [ua+UA_N], rcx
    add [samples], rcx
    lea rcx, [rcx*2+64]
    call mem_alloc
    mov [ua+UA_PCM], rax
    mov rdx, [rsi+CL_DATA]
    mov rcx, [rsi+CL_FRAMES]
    movss xmm1, [c_scale]
    movss xmm2, [c_lo]
    movss xmm3, [c_hi]
    xor r8d, r8d
.s:
    cmp r8, rcx
    jae .converted
    movss xmm0, [rdx+r8*4]
    mulss xmm0, xmm1
    maxss xmm0, xmm2
    minss xmm0, xmm3
    cvtss2si r9d, xmm0
    mov [rax+r8*2], r9w
    inc r8
    jmp .s
.converted:
    mov rcx, [rsi+CL_DATA]
    call mem_free
    mov [ua+UA_ID], rbx
    mov rcx, rbx
.len:
    cmp byte [rcx], 0
    je .counted
    inc rcx
    jmp .len
.counted:
    sub rcx, rbx
    mov [ua+UA_ILEN], ecx
    xor eax, eax
    mov [ua+UA_TEXT], rax
    mov [ua+UA_TLEN], eax
    mov [ua+UA_NORM], rax
    mov [ua+UA_NLEN], eax
    mov [ua+UA_SPK], eax
    mov [ua+UA_ESC], eax
    mov rax, [src]
    mov [ua+UA_SRC], eax
    lea rcx, [sw]
    lea rdx, [ua]
    call sw_add
    mov rcx, [ua+UA_PCM]
    call mem_free
    inc qword [nwav]
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    call lib_init
    ; archives are the args without '='
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
    jz .archive
    cmp al, '='
    je .nextarg
    inc rcx
    jmp .eq
.archive:
    mov rax, [narch]
    lea rdx, [archives]
    mov [rdx+rax*8], rsi
    inc qword [narch]
.nextarg:
    inc rbx
    jmp .arg
.args:
    call cfg_args
    lea rcx, [k_out]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov rbx, rax
    lea rcx, [k_src]
    xor edx, edx
    call cfg_int
    mov [src], rax
    lea rcx, [k_match]
    xor edx, edx
    call cfg_str
    mov [match], rax
    cmp qword [narch], 0
    je .usage
    mov rcx, rbx
    call make_parent
    lea rcx, [sw]
    mov rdx, rbx
    call sw_open
    test eax, eax
    jnz .each
    lea rcx, [e_out]
    call fatal
.each:
    xor edi, edi
.a:
    cmp rdi, [narch]
    jae .close
    lea rax, [archives]
    mov rcx, [rax+rdi*8]
    call file_map
    test rax, rax
    jnz .mapped
    lea rcx, [e_open]
    call fatal
.mapped:
    mov rsi, rax
    cmp dword [rsi], 0x04034b50 ; a zip starts with a local header
    je .zip
    lea rcx, [ta]
    mov r8, rdx
    mov rdx, rsi
    call tar_open
.tm:
    lea rcx, [ta]
    call tar_next
    cmp eax, 1
    jne .nexta
    mov rcx, [ta+TA_NAME]
    call wanted
    test eax, eax
    jz .tm
    mov rcx, [ta+TA_DATA]
    mov rdx, [ta+TA_SIZE]
    mov r8, [ta+TA_NAME]
    call onewav
    jmp .tm
.zip:
    lea rcx, [zp]
    mov r8, rdx
    mov rdx, rsi
    call zip_open
    test eax, eax
    jz .nexta
.zm:
    lea rcx, [zp]
    call zip_next
    test eax, eax
    jz .nexta
    lea rcx, [zp+ZP_PATH]
    call wanted
    test eax, eax
    jz .zm
    mov rcx, [zp+ZP_USIZE]
    add rcx, 64
    call mem_alloc
    mov rsi, rax
    lea rcx, [zp]
    mov rdx, rsi
    call zip_extract
    test eax, eax
    jnz .unzipped
    inc qword [nbad]
    jmp .zfree
.unzipped:
    mov rcx, rsi
    mov rdx, [zp+ZP_USIZE]
    lea r8, [zp+ZP_PATH]
    call onewav
.zfree:
    mov rcx, rsi
    call mem_free
    jmp .zm
.nexta:
    inc rdi
    jmp .a
.close:
    lea rcx, [sw]
    call sw_close
    mov rcx, [nwav]
    call print_dec
    say " wavs, "
    cvtsi2sd xmm0, qword [samples]
    divsd xmm0, [c_rate]
    divsd xmm0, [c_hour]
    mov edx, 2
    call print_fixed
    say " h, "
    mov rcx, [nbad]
    call print_dec
    say " couldn't be read", 13, 10
    call con_restore
    mov rcx, [nbad]
    call ExitProcess
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
