def predict_qe(Co, k_f, inv_n, vm = 0.1, n_iter = 40, damping = 0.3):
    """
    Ce = Co - (qe*vm)
    qe = k_f * Ce ** inv_n
    """
    qe = 0.5 * Co *vm
    for n in range(n_iter):
        Ce = Co - (qe * vm)
        qe_new = k_f * (Ce ** inv_n)
        qe = (1-damping) * qe + damping * qe_new
    return qe

