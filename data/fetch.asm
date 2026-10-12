; fetch URL DEST [head=N]        one file. head=N gets just its first N bytes
; fetch list=FILE [jobs=4]       every "URL DEST" line of a manifest (# starts a comment)
; fetch mdc=ID DEST [key=FILE] [sample=1]
;                                a Mozilla Data Collective dataset: the api is asked for a
;                                signed storage url (with the key in FILE, default
;                                Commonvoiceapikey.txt), then that gets downloaded.
;                                sample=1 gets the dataset's small sample file instead
; files that are already there get skipped. a download lands in DEST.tmp and is
; renamed once it's complete, and a .tmp left by an earlier try gets picked up
; with a Range request, so after a failure just run the same command again.
; started from Mnemonic's data\download.asm
; libs: winhttp.lib
default rel
bits 64
%include "lib.inc"

extern WinHttpOpen, WinHttpConnect, WinHttpOpenRequest, WinHttpSendRequest
extern WinHttpReceiveResponse, WinHttpQueryHeaders, WinHttpReadData, WinHttpCloseHandle
extern ExitProcess, Sleep

BUFSZ   equ 1 << 20
MAXT    equ 64
SECURE  equ 0x00800000

; per thread
T_HOST  equ 0                   ; utf-16, 256 chars
T_PATH  equ 512                 ; utf-16, 4096 chars (signed urls are long)
T_HDR   equ 8704                ; utf-16 extra headers
T_TMP   equ 10752               ; DEST.tmp
T_BUF   equ 11776               ; -> BUFSZ read buffer
T_PORT  equ 11784
T_SEC   equ 11788               ; 1 = https
T_GOT   equ 11792               ; bytes of this file, resumed ones included
T_STAT  equ 11800               ; last http status
T_LEN   equ 11808               ; content length of the last response, -1 if none
T_SIZE  equ 12288

section .rdata
agent   dw __?utf16?__("Mnemonic-voice/1.0"), 0
v_get   dw __?utf16?__("GET"), 0
v_post  dw __?utf16?__("POST"), 0
s_https db "https://", 0
s_http  db "http://", 0
s_tmp   db ".tmp", 0
s_range db "Range: bytes=", 0
s_auth  db "Authorization: Bearer ", 0
mdc_pre db "https://mozilladatacollective.com/api/datasets/", 0
mdc_suf db "/download", 0
mdc_smp db "/sample-download", 0
k_smp   db "sample", 0
j_url   db '"downloadUrl"', 0
j_size  db '"sizeBytes"', 0
j_name  db '"filename"', 0
k_list  db "list", 0
k_jobs  db "jobs", 0
k_head  db "head", 0
k_mdc   db "mdc", 0
k_key   db "key", 0
d_key   db "Commonvoiceapikey.txt", 0
m_got   db "got   ", 0
m_have  db "have  ", 0
m_fail  db "FAILED ", 0
usage   db "usage: fetch URL DEST [head=N]", 13, 10
        db "       fetch list=FILE [jobs=4]", 13, 10
        db "       fetch mdc=DATASET_ID DEST [key=FILE]", 13, 10, 0
e_http  db "WinHttpOpen failed", 0
e_key   db "can't read the api key file", 0
e_list  db "can't read the list", 0
align 8
c_half  dq 0.5
c_mb    dq 1e-6

section .bss
alignb 8
tstate  resb T_SIZE * MAXT
session resq 1
jobs    resq 1                  ; (url, dest) pairs
njobs   resq 1
mdcid   resq 1                  ; the dataset id in mdc mode, else 0
sample  resq 1                  ; mdc: just the dataset's sample file
head    resq 1
done    resq 1
have    resq 1
failed  resq 1
got     resq 1
t0      resq 1
last    resq 1
plock   resd 1
alignb 8
auth    resw 1024               ; "Authorization: Bearer key\r\n" as utf-16
mdcurl  resb 8192
argurl  resb 8192
argdest resb 1024
hdrs    resb 256 * MAXT         ; ascii headers before they get widened
onejob  resq 2

section .text

; rcx = dst, rdx = src. copies a zero terminated string, rax = the dst's 0
copyz:
    mov al, [rdx]
    mov [rcx], al
    test al, al
    jz .end
    inc rcx
    inc rdx
    jmp copyz
.end:
    mov rax, rcx
    ret

; rcx = utf-16 dst, rdx = ascii src
widen:
    movzx eax, byte [rdx]
    mov [rcx], ax
    inc rdx
    add rcx, 2
    test eax, eax
    jnz widen
    ret

; rcx = string, rdx = prefix. eax = 1 if string starts with it
starts:
    mov al, [rdx]
    test al, al
    jz .yes
    cmp al, [rcx]
    jne .no
    inc rcx
    inc rdx
    jmp starts
.yes:
    mov eax, 1
    ret
.no:
    xor eax, eax
    ret

; rcx = path. makes every directory on the way to the file
make_dirs:
    push rbx
    push rsi
    sub rsp, 40
    mov rbx, rcx
    lea rsi, [rcx+1]
.scan:
    mov al, [rsi]
    test al, al
    jz .done
    cmp al, '\'
    je .cut
    cmp al, '/'
    jne .next
.cut:
    mov byte [rsi], 0
    mov [rsp+32], al
    mov rcx, rbx
    call make_dir
    mov al, [rsp+32]
    mov [rsi], al
.next:
    inc rsi
    jmp .scan
.done:
    add rsp, 40
    pop rsi
    pop rbx
    ret

; rcx = url, rdx = thread state. splits it into host, port and path for winhttp.
; eax = 1 if it's an http(s) url
crack:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    mov rbx, rdx
    lea rdx, [s_https]
    call starts
    test eax, eax
    jz .plain
    mov dword [rbx+T_SEC], 1
    mov dword [rbx+T_PORT], 443
    add rsi, 8
    jmp .host
.plain:
    mov rcx, rsi
    lea rdx, [s_http]
    call starts
    test eax, eax
    jz .out
    mov dword [rbx+T_SEC], 0
    mov dword [rbx+T_PORT], 80
    add rsi, 7
.host:
    lea rdi, [rbx+T_HOST]
    mov ecx, 255
.hc:
    movzx eax, byte [rsi]
    test eax, eax
    jz .hostend
    cmp eax, '/'
    je .hostend
    cmp eax, ':'
    je .port
    mov [rdi], ax
    add rdi, 2
    inc rsi
    dec ecx
    jnz .hc
    xor eax, eax
    jmp .out
.port:
    mov word [rdi], 0
    lea rcx, [rsi+1]
    call parse_int
    mov [rbx+T_PORT], eax
    mov rsi, rdx
    jmp .path
.hostend:
    mov word [rdi], 0
.path:
    lea rdi, [rbx+T_PATH]
    mov ecx, 4095
    cmp byte [rsi], 0
    jne .pc
    mov word [rdi], '/'
    add rdi, 2
.pc:
    movzx eax, byte [rsi]
    mov [rdi], ax
    test eax, eax
    jz .ok
    add rdi, 2
    inc rsi
    dec ecx
    jnz .pc
    xor eax, eax
    jmp .out
.ok:
    mov eax, 1
.out:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = path. rax = its size, 0 if it isn't there
size_of:
    push rbx
    sub rsp, 32
    call file_open
    cmp rax, -1
    je .none
    mov rbx, rax
    mov rcx, rax
    call file_size
    mov rcx, rbx
    mov rbx, rax
    call file_close
    mov rax, rbx
    jmp .out
.none:
    xor eax, eax
.out:
    add rsp, 32
    pop rbx
    ret

; locals for request
R_CONN  equ 64
R_REQ   equ 72
R_N     equ 80

; opens a request and sends it. rbx = thread state (cracked url in it), rcx = verb,
; rdx = extra headers (utf-16, 0 = none). rax = request handle, 0 on failure,
; [rsp+...] nothing kept: the connection handle comes back in rdx
request:
    push rbx
    push rsi
    push rdi
    sub rsp, 96
    mov rsi, rcx
    mov rdi, rdx
    xor eax, eax
    mov [rsp+R_CONN], rax
    mov [rsp+R_REQ], rax
    mov rcx, [session]
    lea rdx, [rbx+T_HOST]
    mov r8d, [rbx+T_PORT]
    xor r9d, r9d
    call WinHttpConnect
    test rax, rax
    jz .fail
    mov [rsp+R_CONN], rax
    mov rcx, rax
    mov rdx, rsi
    lea r8, [rbx+T_PATH]
    xor r9d, r9d
    mov qword [rsp+32], 0
    mov qword [rsp+40], 0
    xor eax, eax
    cmp dword [rbx+T_SEC], 0
    je .flags
    mov eax, SECURE
.flags:
    mov [rsp+48], rax
    call WinHttpOpenRequest
    test rax, rax
    jz .fail
    mov [rsp+R_REQ], rax
    mov rcx, rax
    mov rdx, rdi
    xor r8d, r8d
    test rdi, rdi
    jz .send
    mov r8d, -1                 ; zero terminated
.send:
    xor r9d, r9d
    mov qword [rsp+32], 0
    mov qword [rsp+40], 0
    mov qword [rsp+48], 0
    call WinHttpSendRequest
    test eax, eax
    jz .fail
    mov rcx, [rsp+R_REQ]
    xor edx, edx
    call WinHttpReceiveResponse         ; redirects get followed in here
    test eax, eax
    jz .fail
    ; status, and the content length if there is one
    mov dword [rbx+T_STAT], 0
    mov dword [rsp+R_N], 4
    mov rcx, [rsp+R_REQ]
    mov edx, 19 | 0x20000000            ; STATUS_CODE as a number
    xor r8d, r8d
    lea r9, [rbx+T_STAT]
    lea rax, [rsp+R_N]
    mov [rsp+32], rax
    mov qword [rsp+40], 0
    call WinHttpQueryHeaders
    test eax, eax
    jz .fail
    mov qword [rbx+T_LEN], -1
    mov dword [rsp+R_N], 8
    mov rcx, [rsp+R_REQ]
    mov edx, 5 | 0x08000000             ; CONTENT_LENGTH as a 64-bit number
    xor r8d, r8d
    lea r9, [rbx+T_LEN]
    lea rax, [rsp+R_N]
    mov [rsp+32], rax
    mov qword [rsp+40], 0
    call WinHttpQueryHeaders            ; fine if it's missing
    mov rax, [rsp+R_REQ]
    mov rdx, [rsp+R_CONN]
    jmp .out
.fail:
    mov rcx, [rsp+R_REQ]
    test rcx, rcx
    jz .noreq
    call WinHttpCloseHandle
.noreq:
    mov rcx, [rsp+R_CONN]
    test rcx, rcx
    jz .noconn
    call WinHttpCloseHandle
.noconn:
    xor eax, eax
    xor edx, edx
.out:
    add rsp, 96
    pop rdi
    pop rsi
    pop rbx
    ret

; rax = request, rdx = connection
%macro hclose 0
    mov [rsp+40], rdx
    mov rcx, rax
    call WinHttpCloseHandle
    mov rcx, [rsp+40]
    call WinHttpCloseHandle
%endmacro

; one try at url -> the thread's .tmp, resuming what's there.
; rbx = thread state, rcx = url, edx = thread. eax = 1 if the .tmp is complete
attempt:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    ; locals: request +48, connection +56
    mov rsi, rcx
    mov r15d, edx
    xor r13d, r13d              ; request
    mov r14, -1                 ; file
    mov rcx, rsi
    mov rdx, rbx
    call crack
    test eax, eax
    jz .fail
    xor r12d, r12d              ; where this try starts in the file
    cmp qword [head], 0
    jne .hdr
    lea rcx, [rbx+T_TMP]
    call size_of
    mov r12, rax
.hdr:
    ; Range header, if we're resuming or only want the head
    imul rdi, r15, 256
    lea rax, [hdrs]
    add rdi, rax
    mov rcx, rdi
    mov byte [rdi], 0
    mov rax, [head]
    or rax, r12
    jz .noheader
    lea rdx, [s_range]
    call copyz
    mov rcx, rax
    mov rdx, r12
    call fmt_dec
    mov byte [rax], '-'
    inc rax
    mov rcx, rax
    mov rdx, [head]
    test rdx, rdx
    jz .eol
    dec rdx
    call fmt_dec
    mov rcx, rax
.eol:
    mov word [rcx], 0x0a0d
    mov byte [rcx+2], 0
.noheader:
    lea rcx, [rbx+T_HDR]
    mov rdx, rdi
    call widen
    lea rcx, [v_get]
    lea rdx, [rbx+T_HDR]
    cmp byte [rdi], 0
    jne .go
    xor edx, edx
.go:
    call request
    test rax, rax
    jz .fail
    mov r13, rax
    mov [rsp+56], rdx
    mov eax, [rbx+T_STAT]
    cmp eax, 206
    je .open
    cmp eax, 416                ; asked for bytes past the end: the .tmp is already whole
    jne .s200
    test r12, r12
    jz .fail
    mov [rbx+T_GOT], r12
    jmp .whole
.s200:
    cmp eax, 200
    jne .fail
    xor r12d, r12d              ; it ignored the range, start over
.open:
    mov [rbx+T_GOT], r12
    lea rcx, [rbx+T_TMP]
    test r12, r12
    jz .new
    call file_append
    jmp .opened
.new:
    call file_create
.opened:
    cmp rax, -1
    je .fail
    mov r14, rax
    xor edi, edi                ; bytes this try
.read:
    mov dword [rsp+48], 0
    mov rcx, r13
    mov rdx, [rbx+T_BUF]
    mov r8d, BUFSZ
    lea r9, [rsp+48]
    call WinHttpReadData
    test eax, eax
    jz .fail
    mov esi, [rsp+48]
    test esi, esi
    jz .eof
    mov rcx, r14
    mov rdx, [rbx+T_BUF]
    mov r8, rsi
    call file_write
    test eax, eax
    jz .fail
    add rdi, rsi
    add [rbx+T_GOT], rsi
    lock add [got], rsi
    call status
    mov rax, [head]
    test rax, rax
    jz .read
    cmp [rbx+T_GOT], rax
    jb .read
    jmp .whole                  ; a server that ignored the range still stops here
.eof:
    mov rax, [rbx+T_LEN]
    cmp rax, -1
    je .whole
    cmp rdi, rax
    jne .fail                   ; the connection dropped, the .tmp stays for next time
.whole:
    cmp r14, -1
    je .closed
    mov rcx, r14
    call file_flush             ; on the disk before the rename
    test eax, eax
    jz .fail
    mov rcx, r14
    call file_close
    mov r14, -1
.closed:
    mov eax, 1
    jmp .close
.fail:
    xor eax, eax
.close:
    mov [rsp+48], eax
    cmp r14, -1
    je .nofile
    mov rcx, r14
    call file_close
.nofile:
    test r13, r13
    jz .noreq
    mov rax, r13
    mov rdx, [rsp+56]
    hclose
.noreq:
    mov eax, [rsp+48]
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rbx = thread state. asks the data collective for a signed url to dataset mdcid,
; into mdcurl. eax = 1 if it got one
resolve:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 56
    lea rcx, [rbx+T_TMP+512]    ; borrow the end of T_TMP for the api url
    lea rdx, [mdc_pre]
    call copyz
    mov rcx, rax
    mov rdx, [mdcid]
    call copyz
    mov rcx, rax
    lea rdx, [mdc_suf]
    cmp qword [sample], 0
    je .suffix
    lea rdx, [mdc_smp]
.suffix:
    call copyz
    lea rcx, [rbx+T_TMP+512]
    mov rdx, rbx
    call crack
    lea rcx, [v_post]
    lea rdx, [auth]
    call request
    test rax, rax
    jz .fail
    mov r12, rax
    mov [rsp+40], rdx
    ; the reply is small json
    mov rsi, [rbx+T_BUF]
    xor edi, edi
.read:
    mov dword [rsp+32], 0
    mov rcx, r12
    lea rdx, [rsi+rdi]
    mov r8d, 65536
    lea r9, [rsp+32]
    call WinHttpReadData
    test eax, eax
    jz .got
    mov eax, [rsp+32]
    test eax, eax
    jz .got
    add rdi, rax
    cmp rdi, BUFSZ - 65536 - 1
    jb .read
.got:
    mov byte [rsi+rdi], 0
    mov rax, r12
    mov rdx, [rsp+40]
    hclose
    cmp dword [rbx+T_STAT], 200
    je .ok
    say "the data collective said "
    mov ecx, [rbx+T_STAT]
    call print_dec
    say ": "
    cmp rdi, 400
    jbe .show
    mov edi, 400
.show:
    mov rcx, rsi
    mov rdx, rdi
    call print
    say 13, 10
    jmp .fail
.ok:
    mov rcx, rsi
    lea rdx, [j_name]
    lea r8, [mdcurl]
    call jstring
    say "data collective: "
    lea rcx, [mdcurl]
    call print_z
    mov rcx, rsi
    lea rdx, [j_size]
    lea r8, [mdcurl]
    call jstring
    say ", "
    lea rcx, [mdcurl]
    call print_z
    say " bytes", 13, 10
    mov rcx, rsi
    lea rdx, [j_url]
    lea r8, [mdcurl]
    call jstring
    jmp .out
.fail:
    xor eax, eax
.out:
    add rsp, 56
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = json text, rdx = "key" with its quotes, r8 = out. copies the string value
; after the key, undoing \/ \" \\ and \u00xx. eax = 1 if found
jstring:
    push rsi
    push rdi
    push rbx
    mov rsi, rcx
    mov rdi, r8
    mov rbx, rdx
    mov byte [rdi], 0
.find:
    mov rcx, rsi
    mov rdx, rbx
    call starts
    test eax, eax
    jnz .key
    cmp byte [rsi], 0
    je .no
    inc rsi
    jmp .find
.key:
    mov rax, rbx
.klen:
    cmp byte [rax], 0
    je .past
    inc rax
    inc rsi
    jmp .klen
.past:
    mov al, [rsi]
    cmp al, '"'
    je .val
    test al, al
    jz .no
    inc rsi
    jmp .past
.val:
    inc rsi
.ch:
    mov al, [rsi]
    test al, al
    jz .no
    cmp al, '"'
    je .end
    cmp al, '\'
    jne .put
    inc rsi
    mov al, [rsi]
    cmp al, 'u'
    jne .put
    ; \u00xx: only ascii turns up in a url
    movzx eax, byte [rsi+3]
    movzx edx, byte [rsi+4]
    sub eax, '0'
    cmp eax, 9
    jbe .hi
    sub eax, 'a' - '0' - 10
    and eax, 15
.hi:
    sub edx, '0'
    cmp edx, 9
    jbe .lo
    or edx, 0x20
    sub edx, 'a' - '0' - 10
    and edx, 15
.lo:
    shl eax, 4
    or eax, edx
    add rsi, 4
.put:
    mov [rdi], al
    inc rdi
    inc rsi
    jmp .ch
.end:
    mov byte [rdi], 0
    mov eax, 1
    jmp .out
.no:
    xor eax, eax
.out:
    pop rbx
    pop rdi
    pop rsi
    ret

; par_for callback: jobs rdx..r8, r9 = thread
task:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rdx
    mov rdi, r8
    mov rbx, r9
.next:
    cmp rsi, rdi
    jae .done
    mov rcx, rsi
    mov rdx, rbx
    call get_one
    inc rsi
    jmp .next
.done:
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = job, rdx = thread
get_one:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    sub rsp, 40
    mov r14, rdx
    imul rbx, rdx, T_SIZE
    lea rax, [tstate]
    add rbx, rax
    shl rcx, 4
    add rcx, [jobs]
    mov rsi, [rcx]              ; url
    mov rdi, [rcx+8]            ; dest
    mov rcx, rdi
    call file_exists
    test eax, eax
    jz .fetch
    lock inc qword [have]
    lea rcx, [m_have]
    mov rdx, rdi
    mov r8, -1
    call note
    jmp .ret
.fetch:
    mov rcx, rdi
    call make_dirs
    lea rcx, [rbx+T_TMP]
    mov rdx, rdi
    call copyz
    mov rcx, rax
    lea rdx, [s_tmp]
    call copyz
    mov r12d, 5                 ; tries
.try:
    mov rcx, rsi
    mov edx, r14d
    call attempt
    test eax, eax
    jnz .ok
    dec r12d
    jz .bad
    cmp qword [mdcid], 0
    je .wait
    cmp dword [rbx+T_STAT], 403 ; the signed url ran out, get a new one
    jne .wait
    call resolve
.wait:
    mov ecx, 2000
    call Sleep
    jmp .try
.bad:
    lock inc qword [failed]
    lea rcx, [m_fail]
    mov rdx, rdi
    mov r8d, [rbx+T_STAT]
    neg r8
    jnz .shown
    mov r8, -1                  ; never got as far as a status
.shown:
    call note
    jmp .ret
.ok:
    lea rcx, [rbx+T_TMP]
    mov rdx, rdi
    call file_replace
    test eax, eax
    jz .bad
    lock inc qword [done]
    lea rcx, [m_got]
    mov rdx, rdi
    mov r8, [rbx+T_GOT]
    call note
.ret:
    add rsp, 40
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; the one line status, redrawn in place every half second by whoever gets there
status:
    push rbx
    sub rsp, 48
    cmp dword [con_tty], 0
    je .ret
    mov rcx, [last]
    call time_since
    comisd xmm0, [c_half]
    jb .ret
    lock bts dword [plock], 0
    jc .ret                     ; someone else is printing, skip it
    call time_now
    mov [last], rax
    say 13, "  "
    mov rcx, [done]
    add rcx, [failed]
    add rcx, [have]
    call print_dec
    say "/"
    mov rcx, [njobs]
    call print_dec
    say " files   "
    mov rdx, [got]
    lea rcx, [rsp+32]
    call fmt_count
    mov byte [rax], 'B'
    inc rax
    lea rcx, [rsp+32]
    mov rdx, rax
    sub rdx, rcx
    call print
    say " this run   "
    mov rcx, [t0]
    call time_since
    cvtsi2sd xmm1, qword [got]
    divsd xmm1, xmm0
    mulsd xmm1, [c_mb]
    movapd xmm0, xmm1
    mov edx, 1
    call print_fixed
    say " MB/s", 27, "[K"
    mov dword [plock], 0
.ret:
    add rsp, 48
    pop rbx
    ret

; rcx = what happened, rdx = path, r8 = bytes, or -status, or -1. one line above the status
note:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rbx, rcx
    mov rsi, rdx
    mov rdi, r8
.lock:
    lock bts dword [plock], 0
    jnc .locked
    pause
    jmp .lock
.locked:
    cmp dword [con_tty], 0
    je .text
    say 13, 27, "[K"
.text:
    mov rcx, rbx
    call print_z
    mov rcx, rsi
    call print_z
    cmp rdi, -1
    je .eol
    test rdi, rdi
    jns .bytes
    say "  (http "
    mov rcx, rdi
    neg rcx
    call print_dec
    say ")"
    jmp .eol
.bytes:
    say "  "
    mov rcx, rdi
    call print_dec
    say " bytes"
.eol:
    say 13, 10
    mov dword [plock], 0
    mov qword [last], 0         ; redraw the status right away
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret

; rcx = manifest text (cut up in place). fills jobs/njobs
parse_list:
    push rbx
    push rsi
    push rdi
    sub rsp, 32
    mov rsi, rcx
    ; lines over-count jobs, fine for sizing
    xor ebx, ebx
    mov rax, rcx
.count:
    mov dl, [rax]
    test dl, dl
    jz .counted
    cmp dl, 10
    jne .c1
    inc rbx
.c1:
    inc rax
    jmp .count
.counted:
    lea rcx, [rbx*8+rbx]
    lea rcx, [rcx*2+64]         ; 16 bytes a job, plus one
    call mem_alloc
    mov [jobs], rax
    mov rdi, rax
    xor ebx, ebx
.line:
    mov al, [rsi]
    test al, al
    jz .done
    cmp al, ' '
    je .skip1
    cmp al, 9
    je .skip1
    cmp al, 13
    je .skip1
    cmp al, 10
    je .skip1
    cmp al, '#'
    je .comment
    mov [rdi], rsi              ; url
    call .token
    test al, al
    jz .done
    cmp al, 10
    je .line
.gap:
    mov al, [rsi]
    cmp al, ' '
    je .g1
    cmp al, 9
    jne .dest
.g1:
    inc rsi
    jmp .gap
.dest:
    cmp al, 13
    je .line
    cmp al, 10
    je .line
    test al, al
    jz .done
    mov [rdi+8], rsi
    add rdi, 16
    inc rbx
    call .token
    test al, al
    jz .done
    cmp al, 10
    je .line
    ; anything else on the line is ignored
.comment:
    mov al, [rsi]
    test al, al
    jz .done
    inc rsi
    cmp al, 10
    jne .comment
    jmp .line
.skip1:
    inc rsi
    jmp .line
.done:
    mov [njobs], rbx
    add rsp, 32
    pop rdi
    pop rsi
    pop rbx
    ret
; runs rsi to the end of a token and zero terminates it. al = the char that ended it
.token:
    mov al, [rsi]
    test al, al
    jz .tend
    cmp al, ' '
    je .tcut
    cmp al, 9
    je .tcut
    cmp al, 13
    je .tcut
    cmp al, 10
    je .tcut
    inc rsi
    jmp .token
.tcut:
    mov byte [rsi], 0
    inc rsi
.tend:
    ret

global start
start:
    push rbx
    push rsi
    push rdi
    sub rsp, 48
    call lib_init
    cmp qword [argc], 2
    jae .args
.usage:
    lea rcx, [usage]
    call print_z
    mov ecx, 1
    call ExitProcess
.args:
    ; cfg_args writes over every '=', and urls have them, so copy the plain args first
    lea rax, [argv]
    mov rsi, [rax+8]
    mov rcx, rsi
    lea rdx, [s_http]
    call starts
    mov ebx, eax
    mov rcx, rsi
    lea rdx, [s_https]
    call starts
    or ebx, eax
    jz .cfg
    cmp qword [argc], 3
    jb .usage
    lea rcx, [argurl]
    mov rdx, rsi
    call copyz
    lea rax, [argv]
    mov rdx, [rax+16]
    lea rcx, [argdest]
    call copyz
    lea rax, [argurl]
    mov [onejob], rax
    lea rax, [argdest]
    mov [onejob+8], rax
    lea rax, [onejob]
    mov [jobs], rax
    mov qword [njobs], 1
    call cfg_args
    lea rcx, [k_head]
    xor edx, edx
    call cfg_int
    mov [head], rax
    mov ebx, 1
    jmp .pool

.cfg:
    ; mdc=ID DEST: the dest is the one arg without an '='
    lea rax, [argv]
    mov rdx, [rax+16]
    test rdx, rdx
    jz .nodest
    lea rcx, [argdest]
    call copyz
.nodest:
    call cfg_args
    lea rcx, [k_mdc]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .list
    mov [mdcid], rax
    cmp byte [argdest], 0
    je .usage
    lea rcx, [k_smp]
    xor edx, edx
    call cfg_int
    mov [sample], rax
    lea rcx, [k_key]
    lea rdx, [d_key]
    call cfg_str
    mov rcx, rax
    call file_read_all
    test rax, rax
    jnz .key
    lea rcx, [e_key]
    call fatal
.key:
    ; trim the key's trailing whitespace, then "Authorization: Bearer key\r\n"
    mov rsi, rax
.trim:
    test rdx, rdx
    jz .trimmed
    cmp byte [rsi+rdx-1], ' '
    ja .trimmed
    dec rdx
    jmp .trim
.trimmed:
    mov byte [rsi+rdx], 0
    lea rdi, [hdrs]
    mov rcx, rdi
    lea rdx, [s_auth]
    call copyz
    mov rcx, rax
    mov rdx, rsi
    call copyz
    mov word [rax], 0x0a0d
    mov byte [rax+2], 0
    lea rcx, [auth]
    mov rdx, rdi
    call widen
    lea rax, [mdcurl]
    mov [onejob], rax
    lea rax, [argdest]
    mov [onejob+8], rax
    lea rax, [onejob]
    mov [jobs], rax
    mov qword [njobs], 1
    mov ebx, 1
    jmp .pool
.list:
    lea rcx, [k_list]
    xor edx, edx
    call cfg_str
    test rax, rax
    jz .usage
    mov rcx, rax
    call file_read_all
    test rax, rax
    jnz .parse
    lea rcx, [e_list]
    call fatal
.parse:
    mov rcx, rax
    call parse_list
    lea rcx, [k_jobs]
    mov edx, 4
    call cfg_int
    mov ebx, eax
    cmp ebx, MAXT
    jbe .pool
    mov ebx, MAXT

.pool:
    mov ecx, ebx
    call pool_init
    xor esi, esi
.bufs:
    cmp rsi, [nthreads]
    jae .open
    mov ecx, BUFSZ
    call mem_alloc
    imul rcx, rsi, T_SIZE
    lea rdx, [tstate]
    mov [rdx+rcx+T_BUF], rax
    inc esi
    jmp .bufs
.open:
    lea rcx, [agent]
    mov edx, 4                  ; WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY
    xor r8d, r8d
    xor r9d, r9d
    mov qword [rsp+32], 0
    call WinHttpOpen
    test rax, rax
    jnz .session
    lea rcx, [e_http]
    call fatal
.session:
    mov [session], rax
    cmp qword [mdcid], 0
    je .run
    lea rcx, [argdest]
    call file_exists
    test eax, eax
    jnz .run                    ; already have it, don't spend a download on the api
    lea rbx, [tstate]
    call resolve
    test eax, eax
    jnz .run
    mov ecx, 1
    call ExitProcess
.run:
    call time_now
    mov [t0], rax
    lea rcx, [task]
    xor edx, edx
    mov r8, [njobs]
    mov r9d, 1
    call par_for
    mov qword [last], 0
    call status
    say 13, 10
    mov rcx, [done]
    call print_dec
    say " got, "
    mov rcx, [have]
    call print_dec
    say " already there, "
    mov rcx, [failed]
    call print_dec
    say " failed", 13, 10
    mov rcx, [session]
    call WinHttpCloseHandle
    call con_restore
    mov rcx, [failed]
    call ExitProcess
