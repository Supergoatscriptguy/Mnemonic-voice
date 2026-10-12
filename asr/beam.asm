; ctc prefix beam search. prefixes of tokens live in a trie; each node knows its
; words so far for the language model (asr\lm), so a prefix is scored once, when
; it's made. frame by frame the best bm_width prefixes go on, each with the log
; probability of ending in a blank and in its last token. a prefix's score is
; that, plus bm_alpha times its words' language model log probability, plus
; bm_beta per word. single threaded
default rel
bits 64
%include "lib.inc"
%include "tokenizer/tok.inc"
%include "asr/lm.inc"

MAXW     equ 32                 ; widest beam
MAXK     equ 15                 ; most new tokens tried a frame
NSLOT    equ 1024               ; next-frame entries (>= 2 MAXW (MAXK + 1))

; a trie node
ND_PARENT equ 0                 ; u32
ND_TOK    equ 4                 ; i32, -1 at the root
ND_W0     equ 8                 ; the two words before the one being spelled
ND_W1     equ 16
ND_WH     equ 24                ; the word being spelled, its hash so far
ND_INW    equ 32                ; u32 1 while spelling one
ND_NW     equ 36                ; u32 words finished
ND_LM     equ 40                ; f64 their language model log probabilities
ND_SIZE   equ 48

section .rdata
align 8
c_neg    dq -1.0e30
c_far    dq -30.0
c_one    dq 1.0
c_skip   dq -0.0005             ; blank this sure: nothing new this frame
c_keep   dq -10.0               ; new tokens within this of the frame's best

section .data
align 8
global bm_width, bm_topk, bm_alpha, bm_beta, bm_lm, bm_score
bm_width dq 16
bm_topk  dq 8
bm_alpha dq 0.5
bm_beta  dq 1.0
bm_lm    dq 0                   ; an LM (lm_load), or 0 for none
bm_score dq 0                   ; f64 the last decode's best score

section .bss
alignb 8
nodes    resq 1
ncap     resq 1                 ; nodes there's room for
nn       resq 1
ctab     resq 1                 ; children: u64 key (parent << 16 | token) + 1, u32 node
cval     resq 1
cmask    resq 1
lpbase   resq 1
nframes  resq 1
row      resq 1                 ; this frame's log probabilities
lpblank  resq 1                 ; f64
ncur     resq 1
cb_node  resd MAXW
cb_pb    resq MAXW
cb_pnb   resq MAXW
ncand    resq 1
cand     resd MAXK
candlp   resd MAXK
nt_key   resd NSLOT             ; node + 1, 0 = free
nt_pb    resq NSLOT
nt_pnb   resq NSLOT
nt_used  resd NSLOT
nused    resq 1
sc       resq NSLOT             ; scores of the used entries
best     resq 1
toks     resd 8192

section .text

; xmm0 = a, xmm1 = b: xmm0 = ln(e^a + e^b). touches xmm1-xmm3
lse:
    movapd xmm2, xmm0
    maxsd xmm0, xmm1
    minsd xmm1, xmm2
    subsd xmm1, xmm0
    comisd xmm1, [c_far]
    jb .hi
    sub rsp, 40
    movsd [rsp+32], xmm0
    movapd xmm0, xmm1
    call math_exp
    addsd xmm0, [c_one]
    call math_log
    addsd xmm0, [rsp+32]
    add rsp, 40
.hi:
    ret

; rbx = node: the word it was spelling is done. language model, then on to the next
finish:
    sub rsp, 40
    mov rcx, [bm_lm]
    test rcx, rcx
    jz .nolm
    mov rdx, [rbx+ND_W0]
    mov r8, [rbx+ND_W1]
    mov r9, [rbx+ND_WH]
    call lm_score
    addsd xmm0, [rbx+ND_LM]
    movsd [rbx+ND_LM], xmm0
.nolm:
    mov rax, [rbx+ND_W1]
    mov [rbx+ND_W0], rax
    mov rax, [rbx+ND_WH]
    mov [rbx+ND_W1], rax
    inc dword [rbx+ND_NW]
    mov dword [rbx+ND_INW], 0
    call lm_wstart
    mov [rbx+ND_WH], rax
    add rsp, 40
    ret

; ecx = parent, edx = token: eax = the child node, made if it's new
child:
    push rbx
    push rsi
    push rdi
    push r12
    sub rsp, 40
    mov esi, ecx
    mov edi, edx
    mov r12, rsi
    shl r12, 16
    or r12, rdi
    inc r12                     ; the key, never 0
    mov rax, r12
    mov rdx, 0x9e3779b97f4a7c15
    imul rax, rdx
    shr rax, 20
.p:
    and rax, [cmask]
    mov rcx, [ctab]
    mov rdx, [rcx+rax*8]
    test rdx, rdx
    jz .new
    cmp rdx, r12
    je .found
    inc rax
    jmp .p
.found:
    mov rcx, [cval]
    mov eax, [rcx+rax*4]
    jmp .out
.new:
    mov [rcx+rax*8], r12
    mov rdx, [nn]
    mov rcx, [cval]
    mov [rcx+rax*4], edx
    inc qword [nn]
    ; the parent's words, then this token's bytes through the word speller
    imul rbx, rdx, ND_SIZE
    add rbx, [nodes]
    imul rax, rsi, ND_SIZE
    add rax, [nodes]
    mov rcx, [rax+ND_W0]
    mov [rbx+ND_W0], rcx
    mov rcx, [rax+ND_W1]
    mov [rbx+ND_W1], rcx
    mov rcx, [rax+ND_WH]
    mov [rbx+ND_WH], rcx
    mov rcx, [rax+ND_INW]       ; and NW
    mov [rbx+ND_INW], rcx
    mov rcx, [rax+ND_LM]
    mov [rbx+ND_LM], rcx
    mov [rbx+ND_PARENT], esi
    mov [rbx+ND_TOK], edi
    mov [rsp+32], rdx
    mov ecx, edi
    call tok_bytes
    mov rsi, rax
    mov rdi, rdx
.b:
    test rdi, rdi
    jz .made
    movzx ecx, byte [rsi]
    inc rsi
    dec rdi
    lea rax, [lm_wtab]
    movzx ecx, byte [rax+rcx]
    test ecx, ecx
    jz .sep
    mov rax, [rbx+ND_WH]
    xor rax, rcx
    mov rdx, 0x100000001b3
    imul rax, rdx
    mov [rbx+ND_WH], rax
    mov dword [rbx+ND_INW], 1
    jmp .b
.sep:
    cmp dword [rbx+ND_INW], 0
    je .b
    call finish
    jmp .b
.made:
    mov rax, [rsp+32]
.out:
    add rsp, 40
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = node, edx = 0 for its blank ending, 8 for its token ending, xmm0 = a log
; probability: added into the node's entry for the next frame
addn:
    push rbx
    sub rsp, 48
    mov ebx, edx
    lea eax, [rcx+1]
    imul eax, eax, 0x9e3779b1
    shr eax, 22                 ; 10 bits
    lea r8, [nt_key]
.p:
    and eax, NSLOT - 1
    mov edx, [r8+rax*4]
    test edx, edx
    jz .new
    lea r9d, [rcx+1]
    cmp edx, r9d
    je .have
    inc eax
    jmp .p
.new:
    lea edx, [rcx+1]
    mov [r8+rax*4], edx
    mov rdx, [c_neg]
    lea r9, [nt_pb]
    mov [r9+rax*8], rdx
    lea r9, [nt_pnb]
    mov [r9+rax*8], rdx
    mov rdx, [nused]
    lea r9, [nt_used]
    mov [r9+rdx*4], eax
    inc qword [nused]
.have:
    lea r9, [nt_pb]
    test ebx, ebx
    jz .w
    lea r9, [nt_pnb]
.w:
    lea r9, [r9+rax*8]
    movsd xmm1, [r9]
    mov [rsp+32], r9
    call lse
    mov r9, [rsp+32]
    movsd [r9], xmm0
    add rsp, 48
    pop rbx
    ret

; ecx = frames: room for the nodes they can make, the trie emptied
room:
    push rbx
    sub rsp, 32
    mov eax, ecx
    mov rdx, [bm_width]
    imul rax, rdx
    mov rdx, [bm_topk]
    inc rdx
    imul rax, rdx
    add rax, 64
    mov rbx, rax
    cmp rbx, [ncap]
    jbe .clear
    mov rcx, [nodes]
    test rcx, rcx
    jz .na
    call mem_free
    mov rcx, [ctab]
    call mem_free
    mov rcx, [cval]
    call mem_free
.na:
    mov [ncap], rbx
    imul rcx, rbx, ND_SIZE
    call mem_alloc
    mov [nodes], rax
    lea rax, [rbx+rbx]
    bsr rcx, rax
    inc ecx
    mov eax, 1
    shl rax, cl
    lea rdx, [rax-1]
    mov [cmask], rdx
    mov rbx, rax
    lea rcx, [rax*8]
    call mem_alloc
    mov [ctab], rax
    lea rcx, [rbx*4]
    call mem_alloc
    mov [cval], rax
    jmp .out
.clear:
    mov rcx, [ctab]
    mov rdx, [cmask]
    xor eax, eax
.z:
    mov qword [rcx+rax*8], 0
    inc rax
    cmp rax, rdx
    jbe .z
.out:
    add rsp, 32
    pop rbx
    ret

; the frame's new tokens: the bm_topk best that aren't the blank, within c_keep of
; the best of them. none when the blank is all but certain
candidates:
    mov qword [ncand], 0
    movsd xmm0, [lpblank]
    comisd xmm0, [c_skip]
    ja .out
    mov r8, [row]
    mov r9, [bm_topk]
    xor ecx, ecx                ; token
.t:
    cmp ecx, TOK_BLANK
    jae .prune
    movss xmm0, [r8+rcx*4]
    ; insert into the sorted list if it beats its last
    mov rdx, [ncand]
    cmp rdx, r9
    jb .ins
    lea rax, [candlp]
    comiss xmm0, [rax+rdx*4-4]
    jbe .next
    dec rdx
.ins:
    ; shift the worse ones down
.sh:
    test rdx, rdx
    jz .put
    lea rax, [candlp]
    comiss xmm0, [rax+rdx*4-4]
    jbe .put
    movss xmm1, [rax+rdx*4-4]
    movss [rax+rdx*4], xmm1
    lea rax, [cand]
    mov r10d, [rax+rdx*4-4]
    mov [rax+rdx*4], r10d
    dec rdx
    jmp .sh
.put:
    lea rax, [candlp]
    movss [rax+rdx*4], xmm0
    lea rax, [cand]
    mov [rax+rdx*4], ecx
    cmp qword [ncand], r9
    jae .next
    inc qword [ncand]
.next:
    inc ecx
    jmp .t
.prune:
    cmp qword [ncand], 0
    je .out
    movss xmm1, [candlp]
    cvtss2sd xmm1, xmm1
    addsd xmm1, [c_keep]
    mov rdx, [ncand]
.pr:
    lea rax, [candlp]
    movss xmm0, [rax+rdx*4-4]
    cvtss2sd xmm0, xmm0
    comisd xmm0, xmm1
    jae .out
    dec rdx
    mov [ncand], rdx
    jnz .pr
.out:
    ret

; rcx = log probabilities f32 [frames][VOCAB], edx = frames, r8 = out (u16 tokens).
; rax = tokens written
global beam_decode
beam_decode:
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    mov [lpbase], rcx
    mov [nframes], rdx
    mov r15, r8
    mov ecx, edx
    call room
    ; the root
    mov rbx, [nodes]
    mov dword [rbx+ND_PARENT], 0
    mov dword [rbx+ND_TOK], -1
    mov rax, LM_START
    mov [rbx+ND_W0], rax
    mov [rbx+ND_W1], rax
    call lm_wstart
    mov [rbx+ND_WH], rax
    mov qword [rbx+ND_INW], 0
    mov qword [rbx+ND_LM], 0
    mov qword [nn], 1
    mov qword [ncur], 1
    mov dword [cb_node], 0
    mov qword [cb_pb], 0
    mov rax, [c_neg]
    mov [cb_pnb], rax
    xor r12d, r12d              ; frame
.frame:
    cmp r12, [nframes]
    jae .final
    mov rax, r12
    imul rax, VOCAB * 4
    add rax, [lpbase]
    mov [row], rax
    movss xmm0, [rax+TOK_BLANK*4]
    cvtss2sd xmm0, xmm0
    movsd [lpblank], xmm0
    call candidates
    ; empty the next frame's table
    xor ecx, ecx
.cl:
    cmp rcx, [nused]
    jae .cld
    lea rax, [nt_used]
    mov eax, [rax+rcx*4]
    lea rdx, [nt_key]
    mov dword [rdx+rax*4], 0
    inc rcx
    jmp .cl
.cld:
    mov qword [nused], 0
    ; every beam: on with a blank, on with its own token again, and each new one
    xor r13d, r13d
.beam:
    cmp r13, [ncur]
    jae .select
    lea rax, [cb_node]
    mov r14d, [rax+r13*4]       ; node
    lea rax, [cb_pb]
    movsd xmm0, [rax+r13*8]
    lea rax, [cb_pnb]
    movsd xmm1, [rax+r13*8]
    call lse
    movsd [rsp+32], xmm0        ; ptot
    addsd xmm0, [lpblank]
    mov ecx, r14d
    xor edx, edx
    call addn
    imul rbx, r14, ND_SIZE
    add rbx, [nodes]
    mov esi, [rbx+ND_TOK]       ; its last token, -1 at the root
    test esi, esi
    js .new
    lea rax, [cb_pnb]
    movsd xmm0, [rax+r13*8]
    mov rax, [row]
    cvtss2sd xmm1, [rax+rsi*4]
    addsd xmm0, xmm1
    mov ecx, r14d
    mov edx, 8
    call addn
.new:
    xor edi, edi
.c:
    cmp rdi, [ncand]
    jae .beamn
    lea rax, [cand]
    mov edx, [rax+rdi*4]
    mov [rsp+40], rdx
    mov ecx, r14d
    call child
    mov ecx, eax
    mov rdx, [rsp+40]
    movsd xmm0, [rsp+32]        ; from either ending...
    cmp edx, esi
    jne .add
    lea rax, [cb_pb]            ; ...but a repeat needs a blank between
    movsd xmm0, [rax+r13*8]
.add:
    lea rax, [candlp]
    cvtss2sd xmm1, [rax+rdi*4]
    addsd xmm0, xmm1
    mov edx, 8
    call addn
    inc rdi
    jmp .c
.beamn:
    inc r13
    jmp .beam

.select:
    ; scores, then the best bm_width of them on
    xor ecx, ecx
.s:
    cmp rcx, [nused]
    jae .pick
    mov [rsp+40], rcx
    lea rax, [nt_used]
    mov eax, [rax+rcx*4]
    lea rdx, [nt_pb]
    movsd xmm0, [rdx+rax*8]
    lea rdx, [nt_pnb]
    movsd xmm1, [rdx+rax*8]
    lea rdx, [nt_key]
    mov ebx, [rdx+rax*4]
    dec ebx
    call lse
    mov ecx, ebx
    call prefix_score
    mov rcx, [rsp+40]
    lea rax, [sc]
    movsd [rax+rcx*8], xmm0
    inc rcx
    jmp .s
.pick:
    xor r13d, r13d              ; picked
.pk:
    cmp r13, [bm_width]
    jae .picked
    cmp r13, [nused]
    jae .picked
    ; the best of those left
    mov rbx, -1
    movsd xmm1, [c_neg]
    xor ecx, ecx
.f:
    cmp rcx, [nused]
    jae .fd
    lea rax, [sc]
    movsd xmm0, [rax+rcx*8]
    comisd xmm0, xmm1
    jbe .fn
    movapd xmm1, xmm0
    mov rbx, rcx
.fn:
    inc rcx
    jmp .f
.fd:
    test rbx, rbx
    js .picked
    lea rax, [sc]
    mov rdx, [c_neg]
    mov [rax+rbx*8], rdx
    lea rax, [nt_used]
    mov eax, [rax+rbx*4]
    lea rdx, [nt_key]
    mov ecx, [rdx+rax*4]
    dec ecx
    lea rdx, [cb_node]
    mov [rdx+r13*4], ecx
    lea rdx, [nt_pb]
    mov rcx, [rdx+rax*8]
    lea rdx, [cb_pb]
    mov [rdx+r13*8], rcx
    lea rdx, [nt_pnb]
    mov rcx, [rdx+rax*8]
    lea rdx, [cb_pnb]
    mov [rdx+r13*8], rcx
    inc r13
    jmp .pk
.picked:
    mov [ncur], r13
    inc r12
    jmp .frame

.final:
    ; each beam with its last word finished and the end of the utterance scored
    mov qword [best], -1
    movsd xmm0, [c_neg]
    movsd [rsp+32], xmm0
    xor r13d, r13d
.fb:
    cmp r13, [ncur]
    jae .back
    lea rax, [cb_pb]
    movsd xmm0, [rax+r13*8]
    lea rax, [cb_pnb]
    movsd xmm1, [rax+r13*8]
    call lse
    lea rax, [cb_node]
    mov ecx, [rax+r13*4]
    call end_score
    comisd xmm0, [rsp+32]
    jbe .fbn
    movsd [rsp+32], xmm0
    mov [best], r13
.fbn:
    inc r13
    jmp .fb
.back:
    movsd xmm0, [rsp+32]
    movsd [bm_score], xmm0
    ; the best one's tokens, from its node back to the root
    xor eax, eax
    mov r13, [best]
    test r13, r13
    js .done
    lea rax, [cb_node]
    mov ecx, [rax+r13*4]
    xor edi, edi
.up:
    test ecx, ecx
    jz .rev
    imul rbx, rcx, ND_SIZE
    add rbx, [nodes]
    mov eax, [rbx+ND_TOK]
    lea rdx, [toks]
    mov [rdx+rdi*4], eax
    inc rdi
    mov ecx, [rbx+ND_PARENT]
    cmp rdi, 8192
    jb .up
.rev:
    xor eax, eax
.r:
    test rdi, rdi
    jz .done
    dec rdi
    lea rdx, [toks]
    mov ecx, [rdx+rdi*4]
    mov [r15+rax*2], cx
    inc rax
    jmp .r
.done:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret

; ecx = node, xmm0 = its acoustic log probability. xmm0 = its score
prefix_score:
    imul rax, rcx, ND_SIZE
    add rax, [nodes]
    movsd xmm1, [rax+ND_LM]
    mulsd xmm1, [bm_alpha]
    addsd xmm0, xmm1
    cvtsi2sd xmm1, dword [rax+ND_NW]
    mulsd xmm1, [bm_beta]
    addsd xmm0, xmm1
    ret

; ecx = node, xmm0 = acoustic log probability. xmm0 = its score as a whole
; utterance: the word it's spelling finished, then the end. the node's left as is
end_score:
    push rbx
    sub rsp, 48
    movsd [rsp+32], xmm0
    imul rax, rcx, ND_SIZE
    add rax, [nodes]
    ; a scratch copy to finish things on
    lea rbx, [scratch]
    mov rcx, [rax+ND_W0]
    mov [rbx+ND_W0], rcx
    mov rcx, [rax+ND_W1]
    mov [rbx+ND_W1], rcx
    mov rcx, [rax+ND_WH]
    mov [rbx+ND_WH], rcx
    mov rcx, [rax+ND_INW]
    mov [rbx+ND_INW], rcx
    mov rcx, [rax+ND_LM]
    mov [rbx+ND_LM], rcx
    cmp dword [rbx+ND_INW], 0
    je .end
    call finish
.end:
    mov rax, LM_END
    mov [rbx+ND_WH], rax
    mov rcx, [bm_lm]
    test rcx, rcx
    jz .nolm
    mov rdx, [rbx+ND_W0]
    mov r8, [rbx+ND_W1]
    mov r9, LM_END
    call lm_score
    addsd xmm0, [rbx+ND_LM]
    movsd [rbx+ND_LM], xmm0
.nolm:
    movsd xmm0, [rsp+32]
    movsd xmm1, [rbx+ND_LM]
    mulsd xmm1, [bm_alpha]
    addsd xmm0, xmm1
    cvtsi2sd xmm1, dword [rbx+ND_NW]
    mulsd xmm1, [bm_beta]
    addsd xmm0, xmm1
    add rsp, 48
    pop rbx
    ret

section .bss
alignb 8
scratch  resb ND_SIZE
