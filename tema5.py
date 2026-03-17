import numpy as np

def jacobi_eigenvalues(A, epsilon=1e-10, max_iter=10000):
   
    n = A.shape[0]
    A_work = A.astype(float).copy()  
    U = np.eye(n)                    
    
    num_iter = 0
    
    for iteration in range(max_iter):
       
        max_val = 0.0
        p, q = 0, 1
        for i in range(1, n):
            for j in range(0, i):
                if abs(A_work[i, j]) > max_val:
                    max_val = abs(A_work[i, j])
                    p = j  
                    q = i
        
        if max_val < epsilon:
            num_iter = iteration
            break
        
        a_pq = A_work[p, q]
        a_pp = A_work[p, p]
        a_qq = A_work[q, q]
        
        if abs(a_pq) < 1e-15:
            continue
        
        alpha = (a_qq - a_pp) / (2.0 * a_pq)
        
        if alpha >= 0:
            sgn_alpha = 1.0
        else:
            sgn_alpha = -1.0
        
        t = sgn_alpha / (abs(alpha) + np.sqrt(alpha**2 + 1.0))
        
        c = 1.0 / np.sqrt(1.0 + t**2)
        s = c * t
        
        R = np.eye(n)
        R[p, p] = c
        R[q, q] = c
        R[p, q] = s
        R[q, p] = -s
        
        A_work = R.T @ A_work @ R
        
        U = U @ R
        
        num_iter = iteration + 1
    
    eigenvalues = np.diag(A_work)
    eigenvectors = U
    
    return eigenvalues, eigenvectors, num_iter


def verify_jacobi(A_init, eigenvalues, eigenvectors):
    
    U = eigenvectors
    Lambda = np.diag(eigenvalues)
    
    diff = A_init @ U - U @ Lambda
    norm = np.linalg.norm(diff)
    
    return norm


def cholesky_eigenvalue_convergence(A, epsilon=1e-10, max_iter=100):
   
    n = A.shape[0]
    A_k = A.astype(float).copy()
    
    for k in range(max_iter):
        try:
            L = np.linalg.cholesky(A_k)
        except np.linalg.LinAlgError:
            print(f"  [Cholesky] Matricea nu este pozitiv definita la iteratia {k}.")
            print(f"  [Cholesky] Nu se poate continua. Diagonala curenta: {np.diag(A_k)}")
            return np.diag(A_k), k
        
        A_new = L.T @ L
        
        diff = np.linalg.norm(A_new - A_k)
        A_k = A_new
        
        if diff < epsilon:
            break
    
    return np.diag(A_k), k + 1

def svd_analysis(A):
   
    p, n_cols = A.shape
    
    print(f"\n{'='*60}")
    print(f"  ANALIZA SVD - Matrice {p} x {n_cols}")
    print(f"{'='*60}")
    print(f"\nMatricea A:")
    print(A)
    
    U_full, sigma, Vt_full = np.linalg.svd(A, full_matrices=True)
    
    print(f"\n--- Valorile singulare ---")
    print(f"  sigma = {sigma}")
    
    tol = 1e-10
    rank = np.sum(sigma > tol)
    print(f"\n--- Rangul matricei ---")
    print(f"  rang(A) = {rank}")
   
    sigma_nonzero = sigma[sigma > tol]
    if len(sigma_nonzero) > 0:
        cond_number = np.max(sigma_nonzero) / np.min(sigma_nonzero)
    else:
        cond_number = np.inf
    print(f"\n--- Numarul de conditionare ---")
    print(f"  k_2(A) = {cond_number}")
  
    S_pinv = np.zeros((n_cols, p))
    for i in range(len(sigma)):
        if sigma[i] > tol:
            S_pinv[i, i] = 1.0 / sigma[i]
    
   
    V = Vt_full.T
    A_pinv_MP = V @ S_pinv @ U_full.T
    
    print(f"\n--- Pseudoinversa Moore-Penrose (A^I) ---")
    print(A_pinv_MP)
    
    A_pinv_numpy = np.linalg.pinv(A)
    print(f"\n  Verificare cu numpy.linalg.pinv:")
    print(f"  ||A^I_manual - A^I_numpy|| = {np.linalg.norm(A_pinv_MP - A_pinv_numpy)}")
    
    ATA = A.T @ A
    try:
        ATA_inv = np.linalg.inv(ATA)
        A_pinv_LS = ATA_inv @ A.T
        print(f"\n--- Pseudoinversa Least-Squares (A^J) ---")
        print(A_pinv_LS)
    except np.linalg.LinAlgError:
        print(f"\n--- Pseudoinversa Least-Squares (A^J) ---")
        print("  A^T * A nu este inversabila! Nu se poate calcula A^J.")
        A_pinv_LS = None
    
    if A_pinv_LS is not None:
        diff_norm = np.linalg.norm(A_pinv_MP - A_pinv_LS, ord=1)
        print(f"\n--- Norma diferentei ---")
        print(f"  ||A^I - A^J||_1 = {diff_norm}")
    else:
        print(f"\n--- Norma diferentei ---")
        print("  Nu se poate calcula (A^J nu exista).")
    
    return A_pinv_MP, A_pinv_LS

def run_all_tests():
    
    print("=" * 70)
    print("  TEMA 5 - CALCUL NUMERIC")
    print("  Metoda Jacobi + SVD")
    print("=" * 70)
    
    epsilon = 1e-10
    
    test_matrices = []
    
    A1 = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=float)
    test_matrices.append(("Test 1 (3x3)", A1, [-1, 0, 2]))
    
    A2 = np.array([
        [1, 1, 2],
        [1, 1, 2],
        [2, 2, 2]
    ], dtype=float)
    ev2 = [0, 2*(1 - np.sqrt(2)), 2*(1 + np.sqrt(2))]
    test_matrices.append(("Test 2 (3x3)", A2, ev2))

    A3 = np.array([
        [0, 0, 2],
        [0, 1, 0],
        [2, 0, 0]
    ], dtype=float)
    test_matrices.append(("Test 3 (3x3)", A3, [-2, 1, 2]))

    A4 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    test_matrices.append(("Test 4 (4x4)", A4, [0, 0, 2, 2]))

    A5 = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=float)
    ev5 = [0, 0, 2*(4 - np.sqrt(21)), 2*(4 + np.sqrt(21))]
    test_matrices.append(("Test 5 (4x4)", A5, ev5))
    
    for name, A, expected_ev in test_matrices:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"Matricea A:")
        print(A)
        print(f"\nValori proprii asteptate: {sorted(expected_ev)}")
        
        eigenvalues, eigenvectors, num_iter = jacobi_eigenvalues(A, epsilon)

        sorted_ev = np.sort(eigenvalues)
        
        print(f"\n--- Rezultate Metoda Jacobi ---")
        print(f"  Valori proprii calculate: {sorted_ev}")
        print(f"  Numar de iteratii: {num_iter}")

        norm_verif = verify_jacobi(A, eigenvalues, eigenvectors)
        print(f"  Verificare ||A*U - U*Lambda|| = {norm_verif:.2e}")

        ev_numpy = np.sort(np.linalg.eigh(A)[0])
        print(f"\n  Verificare numpy.linalg.eigh: {ev_numpy}")
        print(f"  Diferenta maxima Jacobi vs numpy: {np.max(np.abs(sorted_ev - ev_numpy)):.2e}")

        try:
            np.linalg.cholesky(A)
            is_pos_def = True
        except np.linalg.LinAlgError:
            is_pos_def = False
        
        if is_pos_def:
            print(f"\n--- Convergenta prin Cholesky ---")
            chol_ev, chol_iter = cholesky_eigenvalue_convergence(A, epsilon)
            print(f"  Valori proprii (Cholesky): {np.sort(chol_ev)}")
            print(f"  Iteratii Cholesky: {chol_iter}")
        else:
            print(f"\n--- Convergenta prin Cholesky ---")
            print(f"  Matricea NU este pozitiv definita => Cholesky nu se aplica.")
    
    print(f"\n\n{'#'*70}")
    print(f"  PARTEA 2: SVD - Matrice dreptunghiulare (p > n)")
    print(f"{'#'*70}")

    B1 = np.array([
        [1, 1, 2],
        [1, 1, 2],
        [2, 2, 2],
        [1, 0, 1]
    ], dtype=float)
    svd_analysis(B1)
    
    B2 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ], dtype=float)
    svd_analysis(B2)

    B3 = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ], dtype=float)
    svd_analysis(B3)

if __name__ == "__main__":
    run_all_tests()
