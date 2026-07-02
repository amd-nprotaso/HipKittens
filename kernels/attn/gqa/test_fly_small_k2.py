import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from kernel2 import build_gqa_attn

torch.manual_seed(0)
B, D, H, H_KV, N = 16, 128, 64, 8, 4096
dtype = torch.bfloat16

q = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
k = torch.randn(B, N, H_KV, D, dtype=dtype, device="cuda")
v = torch.randn(B, N, H_KV, D, dtype=dtype, device="cuda")
out = torch.zeros(B, N, H, D, dtype=dtype, device="cuda")
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")

launch = build_gqa_attn(ATTN_B=B, ATTN_H=H, ATTN_H_KV=H_KV, ATTN_N=N, ATTN_D=D)
stream = torch.cuda.current_stream()


def args(q, k, v, out, lse):
    return (
        q.reshape(-1), k.reshape(-1), v.reshape(-1), out.reshape(-1), lse.reshape(-1),
        H * D, H_KV * D, H_KV * D, H * D, N, fx.Stream(stream),
    )


compiled = flyc.compile(launch, *args(q, k, v, out, lse))
print("compiled OK")
compiled(*args(q, k, v, out, lse))
torch.cuda.synchronize()
print("ran OK")
