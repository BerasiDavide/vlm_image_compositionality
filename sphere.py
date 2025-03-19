import torch
#torch.set_grad_enabled(False)

## This code is based on https://github.com/james-oldfield/PoS-subspaces



def orthogonal_projection(s, w, device='cuda'):
    """Orthogonally project the (n+1)-dimensional vectors w onto the tangent space T_sS^n.

    Args:
        s (torch.Tensor): point on S^n
        w (torch.Tensor): batch of (n+1)-dimensional vectors to be projected on T_sS^n

    Returns:
        Pi_s(w) (torch.Tensor): orthogonal projections of w onto T_sS^n

    """
    # Get dimensionality of the ambient space (dim=n+1)
    dim = s.shape[0]

    # Calculate orthogonal projection
    I_ = torch.eye(dim, device=device)
    P = I_ - s.unsqueeze(1) @ s.unsqueeze(1).T

    return w.view(-1, dim) @ P.T


def logarithmic_map(s, q, epsilon=torch.finfo(torch.float32).eps):
    """Calculate the logarithmic map of a batch of sphere points q onto the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points on S^n
        epsilon (uint8) : small value to prevent division by 0

    Returns:
        log_s(q) (torch.Tensor): logarithmic map of q onto the tangent space TsS^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)  # ensure unit length

    pi_s_q_minus_s = orthogonal_projection(s, (q - s), device=q.device)

    return (torch.arccos(torch.clip((q * s).sum(axis=-1), -1.0, 1.0)).unsqueeze(1)) * pi_s_q_minus_s / \
        (torch.norm(pi_s_q_minus_s, p=2, dim=1, keepdim=True) + epsilon)


def exponential_map(s, q):
    """Calculate the exponential map at point s for a batch of points q in the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points in TsS^n.

    Returns:
        exp_s(q) (torch.Tensor): exponential map of q from points in the tangent space TsS^n to S^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension

    q_norm = torch.norm(q, p=2, dim=1).unsqueeze(1)
    out = torch.cos(q_norm) * s + torch.sin(q_norm) * q / q_norm
    return out / torch.norm(out, p=2, dim=-1, keepdim=True)


def weighted_mean(embeddings, weights=None):
    '''Calculate the weighted mean of `embeddings`. The `weights` are normalized.'''
    if weights is None:
        return embeddings.mean(dim=0)
    else:
        return weights @ embeddings / weights.sum()
        

def calculate_intrinstic_mean(data, weights=None, lr=1.000, init=None, eps=1e-5, max_iter=10):
    """Calculate the weighted intrinsic mean of `embeddings`. The `weights` are normalized.""" 
    if data.shape[0]==1:
      return data.squeeze()
    if init is None:
        mean = data[0] # init with first datapoint if not specified
    elif init == 'normalized mean':
        mean = weighted_mean(data, weights) / torch.norm(weighted_mean(data, weights), p=2, dim=-1)
    else:
        mean = init

    update_norm = 1.
    i = 0
    with torch.no_grad():
        while update_norm >= eps and i<max_iter:
            grad = weighted_mean(logarithmic_map(mean, data), weights) # Arithmetic weighted mean
            mean = exponential_map(mean, lr * grad).squeeze()
            i+=1
            update_norm = torch.norm(lr * grad)
    if i == max_iter:
        print(f'Warning! Desired precision not reached after all iterations. update_norm={update_norm}')
    return mean / torch.norm(mean, p=2)


def parallel_transport(p, q, u_in_Tp, v=None):
    '''Transports the vectors `u_in_Tp` from TpS^n to TqS^n. If v=Log_p(q) is provided, q is not used.'''
    # From https://proceedings.neurips.cc/paper/2021/file/1680e9fa7b4dd5d62ece800239bb53bd-Supplemental.pdf RK: formula contains an error: cos(x-1)->cos(x)-1
    
    dim = p.shape[0]
    device = p.device


    if v is None:
        v = logarithmic_map(p, q.unsqueeze(0))  # q = exp_p(v)
    norm_v = torch.linalg.vector_norm(v, ord=2)

    I_ = torch.eye(dim, device=device)
    vTv = v.T @ v
    vTp = v.T @ p.unsqueeze(0)

    P = I_ + (torch.cos(norm_v) - 1) * vTv / norm_v**2 - \
        torch.sin(norm_v) * vTp / norm_v

    u_in_Tq = u_in_Tp @ P.T
    return u_in_Tq