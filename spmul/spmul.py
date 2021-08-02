import torch
import spmul_cuda # use this line if the .cu is already compiled by python setup.py install
# use the following two lines for online compiling
# from torch.utils.cpp_extension import load
# spmul_cuda = load('spmul_cuda', ['spmul_cuda.cu'], verbose=True)


def get_offsets(n_link_all):
    return torch.tensor([0] + [2 ** k for k in range(n_link_all - 1)], dtype=int, requires_grad=False).cuda()


class SparseMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, F, V, offsets=None, n_block=16, n_thread_vec=64, n_thread_dim=16, n_thread_link=16):
        if offsets is None:
            offsets = get_offsets(F.shape[-1])
        ctx.offsets = offsets
        ctx.F = F
        ctx.V = V
        ctx.n_block = n_block
        ctx.n_thread_vec = n_thread_vec
        ctx.n_thread_dim = n_thread_dim
        ctx.n_thread_link = n_thread_link
        return spmul_cuda.forward_host(F, V, offsets, n_block, n_thread_vec, n_thread_dim)

    @staticmethod
    def backward(ctx, dJdZ):
        dJdF, dJdV = spmul_cuda.backward_host(dJdZ, ctx.F, ctx.V, ctx.offsets, ctx.n_block, ctx.n_thread_vec,
                                              ctx.n_thread_dim, ctx.n_thread_link)
        return dJdF, dJdV, None, None, None, None, None
