; stage 0: toolchain + lib linked in, and a beep so it does something you can hear
default rel
bits 64
%include "lib.inc"

extern Beep, ExitProcess

section .text
global start
start:
    sub rsp, 40
    call lib_init
    say "Mnemonic-voice online. x86-64, no CRT, just kernel32.", 13, 10
    ; Beep goes through the default output device on anything newer than win7
    mov ecx, 660
    mov edx, 120
    call Beep
    mov ecx, 880
    mov edx, 180
    call Beep
    call con_restore
    xor ecx, ecx
    call ExitProcess
