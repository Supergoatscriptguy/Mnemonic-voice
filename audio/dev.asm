; audio devices through wasapi: the default render or capture endpoint, shared mode,
; event driven, in whatever format the engine mixes in (float32, in practice always)
default rel
bits 64
%include "lib.inc"
%include "audio/audio.inc"

extern CoInitializeEx, CoCreateInstance, CoTaskMemFree, PropVariantClear
extern CreateEventA, WaitForSingleObject, CloseHandle, WideCharToMultiByte, ExitProcess

CLSCTX_ALL      equ 0x17
EVENTCALLBACK   equ 0x40000
LOOPBACK        equ 0x20000
VT_LPWSTR       equ 31

%macro guid 11
    dd %1
    dw %2, %3
    db %4, %5, %6, %7, %8, %9, %10, %11
%endmacro

; com call: this goes in rcx, methods are numbered from the vtable start
; (IUnknown's QueryInterface, AddRef, Release are 0-2)
%macro vcall 2
    mov rcx, %1
    mov rax, [rcx]
    call [rax + %2*8]
%endmacro

%macro release 1
    mov rcx, %1
    test rcx, rcx
    jz %%no
    mov rax, [rcx]
    call [rax + 16]
%%no:
%endmacro

; a failed hresult ends the program, for now
%macro HR 1+
    test eax, eax
    jns %%ok
    section .rdata
    %%s db %1, 0
    section .text
    lea rcx, [%%s]
    mov edx, eax
    call hr_fail
%%ok:
%endmacro

section .rdata
align 4
CLSID_MMDeviceEnumerator: guid 0xbcde0395, 0xe52f, 0x467c, 0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e
IID_IMMDeviceEnumerator:  guid 0xa95664d2, 0x9614, 0x4f35, 0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6
IID_IAudioClient:         guid 0x1cb9ad4c, 0xdbfa, 0x4c32, 0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2
IID_IAudioRenderClient:   guid 0xf294acfc, 0x3146, 0x4483, 0xa7, 0xbf, 0xad, 0xdc, 0xa7, 0xc2, 0x60, 0xe2
IID_IAudioCaptureClient:  guid 0xc8adbd64, 0xe71e, 0x48a0, 0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17
PKEY_FriendlyName:        guid 0xa45c254e, 0xdf1c, 0x4efd, 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0
                          dd 14
align 8
c_ms    dq 1e-4                 ; 100 ns units -> ms
c_1000  dq 1000.0

section .text

; rcx = what failed, edx = hresult
hr_fail:
    and rsp, -16
    sub rsp, 48
    mov [rsp+32], rcx
    mov [rsp+40], rdx
    say 13, 10, "error: "
    mov rcx, [rsp+32]
    call print_z
    say " failed (hresult 0x"
    mov ecx, [rsp+40]
    mov edx, 8
    call print_hex
    say ")", 13, 10
    call con_restore
    mov ecx, 1
    call ExitProcess

; ecx = 0 render / 1 capture / 2 loopback (capture what the output plays),
; rdx = dev (zeroed), r8d = buffer length in ms.
; opens the default endpoint and gets the stream ready to start.
; eax = 1, or 0 if the mix format isn't float32
global dev_open
dev_open:
    push rbx
    push rsi
    push rdi
    sub rsp, 112                ; room for 8 args, locals from +64
    mov ebx, ecx
    mov rsi, rdx
    mov edi, r8d
    mov [rsi+DV_DIR], ebx
    xor ecx, ecx
    xor edx, edx                ; COINIT_MULTITHREADED
    call CoInitializeEx

    lea rcx, [CLSID_MMDeviceEnumerator]
    xor edx, edx
    mov r8d, CLSCTX_ALL
    lea r9, [IID_IMMDeviceEnumerator]
    lea rax, [rsi+DV_ENUM]
    mov [rsp+32], rax
    call CoCreateInstance
    HR "creating the device enumerator"

    mov edx, ebx                ; eRender / eCapture
    cmp edx, 2
    jne .flow
    xor edx, edx                ; loopback records what the render endpoint plays
.flow:
    xor r8d, r8d                ; eConsole
    lea r9, [rsi+DV_DEVICE]
    vcall [rsi+DV_ENUM], 4      ; GetDefaultAudioEndpoint
    HR "finding the default device"

    ; the friendly name, just to show which device it is
    xor edx, edx                ; STGM_READ
    lea r8, [rsp+64]
    vcall [rsi+DV_DEVICE], 4    ; OpenPropertyStore
    HR "opening the device properties"
    xor eax, eax
    mov [rsp+72], rax           ; PROPVARIANT
    mov [rsp+80], rax
    lea rdx, [PKEY_FriendlyName]
    lea r8, [rsp+72]
    vcall [rsp+64], 5           ; GetValue
    cmp word [rsp+72], VT_LPWSTR
    jne .noname
    mov ecx, 65001              ; utf-8
    xor edx, edx
    mov r8, [rsp+80]
    mov r9d, -1
    lea rax, [rsi+DV_NAME]
    mov [rsp+32], rax
    mov qword [rsp+40], DV_NAMELEN - 1
    mov qword [rsp+48], 0
    mov qword [rsp+56], 0
    call WideCharToMultiByte
.noname:
    lea rcx, [rsp+72]
    call PropVariantClear
    release [rsp+64]

    lea rdx, [IID_IAudioClient]
    mov r8d, CLSCTX_ALL
    xor r9d, r9d
    lea rax, [rsi+DV_CLIENT]
    mov [rsp+32], rax
    vcall [rsi+DV_DEVICE], 3    ; Activate
    HR "activating the audio client"
    lea rdx, [rsi+DV_FMT]
    vcall [rsi+DV_CLIENT], 8    ; GetMixFormat
    HR "getting the mix format"
    mov rax, [rsi+DV_FMT]
    movzx ecx, word [rax+2]
    mov [rsi+DV_CH], ecx
    mov ecx, [rax+4]
    mov [rsi+DV_RATE], ecx
    movzx ecx, word [rax]
    cmp ecx, 0xfffe
    jne .tag
    movzx ecx, word [rax+24]    ; extensible: the subformat guid's first word
.tag:
    cmp ecx, 3
    jne .notfloat
    cmp word [rax+14], 32
    jne .notfloat

    lea rdx, [rsi+DV_PERIOD]
    lea r8, [rsp+88]
    vcall [rsi+DV_CLIENT], 9    ; GetDevicePeriod
    HR "getting the device period"

    xor edx, edx                ; AUDCLNT_SHAREMODE_SHARED
    mov r8d, EVENTCALLBACK
    mov eax, EVENTCALLBACK | LOOPBACK
    cmp ebx, 2
    cmove r8d, eax
    imul r9, rdi, 10000         ; ms -> 100 ns
    mov qword [rsp+32], 0       ; periodicity has to be 0 in shared mode
    mov rax, [rsi+DV_FMT]
    mov [rsp+40], rax
    mov qword [rsp+48], 0
    vcall [rsi+DV_CLIENT], 3    ; Initialize
    HR "initializing the stream"

    xor ecx, ecx
    xor edx, edx                ; auto reset
    xor r8d, r8d
    xor r9d, r9d
    call CreateEventA
    mov [rsi+DV_EVENT], rax
    mov rdx, rax
    vcall [rsi+DV_CLIENT], 13   ; SetEventHandle
    HR "setting the event"
    lea rdx, [rsi+DV_BUF]
    vcall [rsi+DV_CLIENT], 4    ; GetBufferSize
    HR "getting the buffer size"
    lea rdx, [IID_IAudioRenderClient]
    test ebx, ebx
    jz .svc
    lea rdx, [IID_IAudioCaptureClient]
.svc:
    lea r8, [rsi+DV_SVC]
    vcall [rsi+DV_CLIENT], 14   ; GetService
    HR "getting the render/capture client"
    mov eax, 1
    jmp .out
.notfloat:
    xor eax, eax
.out:
    add rsp, 112
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = dev. one line: name, format, buffer and period
global dev_show
dev_show:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    lea rcx, [rbx+DV_NAME]
    call print_z
    say ": "
    mov ecx, [rbx+DV_RATE]
    call print_dec
    say " Hz, "
    mov ecx, [rbx+DV_CH]
    call print_dec
    say " ch float32, buffer "
    mov ecx, [rbx+DV_BUF]
    call print_dec
    say " frames = "
    cvtsi2sd xmm0, dword [rbx+DV_BUF]
    cvtsi2sd xmm1, dword [rbx+DV_RATE]
    divsd xmm0, xmm1
    mulsd xmm0, [c_1000]
    mov edx, 1
    call print_fixed
    say " ms, period "
    cvtsi2sd xmm0, qword [rbx+DV_PERIOD]
    mulsd xmm0, [c_ms]
    mov edx, 1
    call print_fixed
    say " ms", 13, 10
    add rsp, 32
    pop rbx
    ret

; rcx = dev
global dev_start
dev_start:
    sub rsp, 40
    vcall [rcx+DV_CLIENT], 10   ; Start
    HR "starting the stream"
    add rsp, 40
    ret

global dev_stop
dev_stop:
    sub rsp, 40
    vcall [rcx+DV_CLIENT], 11   ; Stop
    HR "stopping the stream"
    add rsp, 40
    ret

global dev_close
dev_close:
    push rbx
    sub rsp, 32
    mov rbx, rcx
    release [rbx+DV_SVC]
    release [rbx+DV_CLIENT]
    release [rbx+DV_DEVICE]
    release [rbx+DV_ENUM]
    mov rcx, [rbx+DV_FMT]
    call CoTaskMemFree
    mov rcx, [rbx+DV_EVENT]
    test rcx, rcx
    jz .out
    call CloseHandle
.out:
    add rsp, 32
    pop rbx
    ret

; rcx = dev. waits for the engine's next period. eax = 0, or 0x102 if 2 s went by
global dev_wait
dev_wait:
    sub rsp, 40
    mov rcx, [rcx+DV_EVENT]
    mov edx, 2000
    call WaitForSingleObject
    add rsp, 40
    ret

; rcx = dev. eax = frames the render buffer can take right now
global dev_space
dev_space:
    push rbx
    sub rsp, 48
    mov rbx, rcx
    lea rdx, [rsp+40]
    vcall [rbx+DV_CLIENT], 6    ; GetCurrentPadding
    HR "GetCurrentPadding"
    mov eax, [rbx+DV_BUF]
    sub eax, [rsp+40]
    add rsp, 48
    pop rbx
    ret

; rcx = dev, rdx = where captured frames go, r8 = room there in frames.
; takes every whole packet that's waiting and fits. rax = frames.
; DV_GLITCH counts packets that came after a gap (the first one always says so)
global dev_read
dev_read:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    sub rsp, 80                 ; GetBuffer has 6 args, locals from +48
    mov rbx, rcx
    mov rdi, rdx
    mov r12, r8
    xor r13d, r13d
.packet:
    lea rdx, [rsp+64]
    vcall [rbx+DV_SVC], 5       ; GetNextPacketSize
    HR "GetNextPacketSize"
    mov eax, [rsp+64]
    test eax, eax
    jz .done
    mov rcx, r12
    sub rcx, r13
    cmp rax, rcx
    ja .done                    ; doesn't fit, leave it for next time
    lea rdx, [rsp+48]           ; data
    lea r8, [rsp+56]            ; frames
    lea r9, [rsp+60]            ; flags
    mov qword [rsp+32], 0
    mov qword [rsp+40], 0
    vcall [rbx+DV_SVC], 3       ; GetBuffer
    HR "capture GetBuffer"
    test dword [rsp+60], 1      ; AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY
    jz .cont
    cmp dword [rbx+DV_PACKETS], 0
    je .cont
    inc dword [rbx+DV_GLITCH]
.cont:
    inc dword [rbx+DV_PACKETS]
    mov ecx, [rsp+56]
    imul ecx, [rbx+DV_CH]
    test dword [rsp+60], 2      ; AUDCLNT_BUFFERFLAGS_SILENT: the data is junk, write zeros
    jnz .silent
    mov rsi, [rsp+48]
    rep movsd
    jmp .release
.silent:
    inc dword [rbx+DV_SILENT]
    xor eax, eax
    rep stosd
.release:
    mov edx, [rsp+56]
    add r13, rdx
    vcall [rbx+DV_SVC], 4       ; ReleaseBuffer
    HR "capture ReleaseBuffer"
    jmp .packet
.done:
    mov rax, r13
    add rsp, 80
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = dev, rdx = f32 frames in the device's channel layout, r8d = how many (<= dev_space)
global dev_write
dev_write:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    mov rbx, rcx
    mov rsi, rdx
    mov [rsp+40], r8
    mov edx, r8d
    lea r8, [rsp+32]
    vcall [rbx+DV_SVC], 3       ; GetBuffer
    HR "GetBuffer"
    mov rdi, [rsp+32]
    mov ecx, [rsp+40]
    imul ecx, [rbx+DV_CH]
    rep movsd
    mov edx, [rsp+40]
    xor r8d, r8d
    vcall [rbx+DV_SVC], 4       ; ReleaseBuffer
    HR "ReleaseBuffer"
    add rsp, 48
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = dev (render, open, not started), rdx = clip at the device's rate.
; lays the clip out in the device's channels (mono goes to all of them, extra
; channels get dropped, missing ones stay silent), plays it and waits for the end.
; eax = 1, or 0 if the device stopped asking for audio
global dev_play
dev_play:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov rbx, rcx
    mov r13, [rdx+CL_FRAMES]
    mov r14d, [rbx+DV_CH]
    mov ecx, [rdx+CL_CH]
    mov [rsp+32], rcx
    mov rcx, [rdx+CL_DATA]
    mov [rsp+40], rcx
    mov rcx, r13
    imul rcx, r14
    lea rcx, [rcx*4+64]
    call mem_alloc
    mov r15, rax
    mov rsi, [rsp+40]
    mov rdi, rax
    mov r9, [rsp+32]
    xor r8d, r8d
    test r13, r13
    jz .laidout
.frame:
    xor ecx, ecx
.ch:
    xor eax, eax
    mov edx, ecx
    cmp r9d, 1
    jne .multi
    xor edx, edx
.multi:
    cmp edx, r9d
    jae .put
    mov eax, [rsi+rdx*4]
.put:
    mov [rdi], eax
    add rdi, 4
    inc ecx
    cmp ecx, r14d
    jb .ch
    lea rsi, [rsi+r9*4]
    inc r8
    cmp r8, r13
    jb .frame
.laidout:
    xor r12d, r12d
    call .feed                  ; fill the buffer before starting, or the first period glitches
    mov rcx, rbx
    call dev_start
.more:
    cmp r12, r13
    jae .drain
    mov rcx, rbx
    call dev_wait
    test eax, eax
    jnz .dead
    call .feed
    jmp .more
.drain:
    mov rcx, rbx
    call dev_wait
    test eax, eax
    jnz .dead
    mov rcx, rbx
    call dev_space
    cmp eax, [rbx+DV_BUF]
    jb .drain
    mov rcx, rbx
    call dev_wait               ; one more period, the engine reads ahead of the speaker
    mov rcx, rbx
    call dev_stop
    mov eax, 1
    jmp .out
.dead:
    xor eax, eax
.out:
    mov [rsp+32], eax
    mov rcx, r15
    call mem_free
    mov eax, [rsp+32]
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; hands the device as much as it'll take: rbx = dev, r12 = frames handed over,
; r13 = all frames, r14 = channels, r15 = samples
.feed:
    push rsi
    sub rsp, 32
    mov rcx, rbx
    call dev_space
    mov rsi, r13
    sub rsi, r12
    cmp rsi, rax
    cmova rsi, rax
    test rsi, rsi
    jz .fed
    mov rdx, r12
    imul rdx, r14
    lea rdx, [r15+rdx*4]
    mov rcx, rbx
    mov r8d, esi
    call dev_write
    add r12, rsi
.fed:
    add rsp, 32
    pop rsi
    ret
