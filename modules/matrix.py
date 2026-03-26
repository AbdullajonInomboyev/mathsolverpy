"""
matrix.py — Matritsa moduli (n×m qo'llab-quvvatlaydi)
Determinant, teskari matritsa, Gauss metodi — linsolve bilan to'liq.
"""
import sympy as sp
from sympy import (Matrix, symbols, latex as sp_latex, eye, Rational,
                   linsolve)
from core.formatter import format_step, build_solution


# ─────────────────────────────────────────────────────────────
# Yordamchi: matritsani chiroyli ko'rsatish
# ─────────────────────────────────────────────────────────────
def _mat_str(M: Matrix) -> str:
    rows = []
    for i in range(M.rows):
        row = "  [ " + "  ".join(str(M[i,j]).rjust(8) for j in range(M.cols)) + " ]"
        rows.append(row)
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────
# 1. DETERMINANT
# ─────────────────────────────────────────────────────────────
def compute_determinant(matrix_data: list) -> dict:
    steps = []
    try:
        M = Matrix(matrix_data)
        n = M.shape[0]
        if M.shape[0] != M.shape[1]:
            return {"error": "Determinant faqat kvadrat matritsa uchun hisoblanadi",
                    "steps": [], "problem_type": "Determinant"}

        steps.append(format_step(1, "Berilgan matritsa",
            f"{n}×{n} matritsa:\n{_mat_str(M)}", M))

        if n == 1:
            det = M[0,0]
            steps.append(format_step(2, "1×1 determinant",
                f"det(A) = {det}", det))

        elif n == 2:
            a,b,c,d = M[0,0],M[0,1],M[1,0],M[1,1]
            det = a*d - b*c
            steps.append(format_step(2, "2×2 Determinant formulasi",
                f"det(A) = ad - bc",
                sp.Eq(sp.Symbol('det'), sp.sympify(f"({a})*({d}) - ({b})*({c})"))))
            steps.append(format_step(3, "Hisoblash",
                f"det = ({a})·({d}) − ({b})·({c}) = {a*d} − {b*c} = {det}",
                sp.Integer(det)))

        elif n == 3:
            steps.append(format_step(2, "Kofaktor yoyilmasi (birinchi qator bo'yicha)",
                "det(A) = a₁₁·C₁₁ + a₁₂·C₁₂ + a₁₃·C₁₃"))
            a11,a12,a13 = M[0,0],M[0,1],M[0,2]
            M11,M12,M13 = (M.minor_submatrix(0,j) for j in range(3))
            c11,c12,c13 = M11.det(), -M12.det(), M13.det()
            steps.append(format_step(3, "Minorlar va kofaktorlar",
                f"C₁₁ = +det(M₁₁) = {c11}\n"
                f"C₁₂ = −det(M₁₂) = {c12}\n"
                f"C₁₃ = +det(M₁₃) = {c13}"))
            det = a11*c11 + a12*c12 + a13*c13
            steps.append(format_step(4, "Determinant qiymati",
                f"det = ({a11})·({c11}) + ({a12})·({c12}) + ({a13})·({c13}) = {det}",
                sp.Integer(det)))

        else:
            det = M.det()
            steps.append(format_step(2, f"{n}×{n} — LU yoyilmasi",
                f"Katta matritsalar uchun LU decomposition ishlatiladi.",
                sp.Integer(det)))

        steps.append(format_step(len(steps)+1, "Xulosa",
            f"det(A) = {det}\n" + ('Qaytariluvchan ✓' if det != 0 else 'Singulyar — teskari YOQ')))

        return build_solution("Determinant", steps, sp.Integer(det),
            f"{'Qaytariluvchan' if det != 0 else 'Singulyar'} matritsa")
    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Determinant"}


# ─────────────────────────────────────────────────────────────
# 2. TESKARI MATRITSA
# ─────────────────────────────────────────────────────────────
def compute_inverse(matrix_data: list) -> dict:
    steps = []
    try:
        M = Matrix(matrix_data)
        n = M.shape[0]
        if M.shape[0] != M.shape[1]:
            return {"error": "Teskari matritsa faqat kvadrat matritsa uchun hisoblanadi",
                    "steps": [], "problem_type": "Teskari matritsa"}

        steps.append(format_step(1, "Berilgan matritsa",
            f"{n}×{n} matritsa:\n{_mat_str(M)}", M))

        det = M.det()
        steps.append(format_step(2, "Determinantni tekshirish",
            f"det(A) = {det}\n" + ('det != 0 → teskari mavjud ✓' if det != 0 else 'det = 0 → teskari YOQ'),

            sp.Eq(sp.Symbol(r'\det(A)'), sp.Integer(det))))

        if det == 0:
            return build_solution("Teskari matritsa", steps, None,
                "Singulyar matritsa — teskari mavjud emas")

        steps.append(format_step(3, "Joriy–Gauss usuli",
            "Kengaytirilgan matritsa: [A | I] → elementar amallar → [I | A⁻¹]"))

        aug = M.row_join(eye(n))
        steps.append(format_step(4, "Kengaytirilgan matritsa [A | I]",
            _mat_str(aug)))

        aug_rref, _ = aug.rref()
        A_inv = aug_rref[:, n:]
        steps.append(format_step(5, "RREF dan teskari matritsa",
            f"A⁻¹:\n{_mat_str(A_inv)}", A_inv))

        check = sp.simplify(M * A_inv)
        steps.append(format_step(6, "Tekshirish: A · A⁻¹ = I",
            f"{'A · A⁻¹ = I ✓' if check == eye(n) else 'Farq bor ⚠'}",
            check))

        return build_solution("Teskari matritsa", steps, A_inv,
            f"det(A) = {det}")
    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Teskari matritsa"}


# ─────────────────────────────────────────────────────────────
# 3. Ax = b SISTEMASI  (m×n matritsa — to'liq qo'llab-quvvatlaydi)
# ─────────────────────────────────────────────────────────────
def solve_matrix_system(A_data: list, b_data: list) -> dict:
    """
    Ax = b sistemasini yechadi.
    • Kvadrat (m=n): yagona yechim
    • Ortiqcha aniqlanган (m>n): mos yechim bo'lsa topadi
    • Kam aniqlanган (m<n): umumiy yechim (erkin o'zgaruvchilar bilan)
    """
    steps = []
    try:
        A = Matrix(A_data)
        b_raw = b_data
        # b: list of lists yoki list of scalars
        if b_raw and isinstance(b_raw[0], list):
            b_list = [row[0] for row in b_raw]
        else:
            b_list = list(b_raw)
        b = Matrix(b_list)

        m, n = A.shape
        steps.append(format_step(1, "Berilgan sistema Ax = b",
            f"A — {m}×{n} matritsa:\n{_mat_str(A)}\n\nb vektori: {b_list}",
            sp.Eq(sp.MatrixSymbol('A',m,n) * sp.MatrixSymbol('x',n,1),
                  sp.MatrixSymbol('b',m,1))))

        # Kengaytirilgan matritsa
        aug = A.row_join(b)
        steps.append(format_step(2, "Kengaytirilgan matritsa [A | b]",
            _mat_str(aug)))

        aug_rref, pivots = aug.rref()
        steps.append(format_step(3, "Gauss — qator ustida amallar (RREF)",
            _mat_str(aug_rref)))

        # Izchillik tekshirish
        rank_A   = A.rank()
        rank_aug = aug.rank()
        steps.append(format_step(4, "Rang tekshiruvi",
            f"rang(A) = {rank_A},  rang([A|b]) = {rank_aug}\n"
            + ("Yagona yechim ✓" if rank_A == rank_aug == n
               else "Ko'p yechim mavjud (erkin o'zgaruvchilar)" if rank_A == rank_aug
               else "Yechim yo'q (mos kelmaydigan sistema) ✗")))

        if rank_A != rank_aug:
            return build_solution("Gauss metodi", steps, None,
                "Mos kelmaydigan sistema — yechim yo'q")

        # o'zgaruvchilarni nomlash
        var_names = [sp.Symbol(f"x{i+1}") for i in range(n)]
        sol_set   = linsolve((A, b), var_names)

        if not sol_set:
            return build_solution("Gauss metodi", steps, None, "Yechim topilmadi")

        sol_tuple = list(sol_set)[0]
        sol_dict  = {var_names[i]: sol_tuple[i] for i in range(n)}

        result_str = "\n".join(f"x{i+1} = {sol_tuple[i]}" for i in range(n))
        steps.append(format_step(5, "Yechim",
            result_str))

        # Tekshirish
        x_vec = Matrix([sol_tuple[i] for i in range(n)])
        check = sp.simplify(A * x_vec - b)
        is_zero = all(v == 0 for v in check)
        steps.append(format_step(6, "Tekshirish: Ax − b = 0",
            f"{'Ax − b = 0 ✓' if is_zero else 'Farq bor ⚠'}",
            check))

        return build_solution("Gauss metodi", steps, sol_dict,
            f"{m}×{n} sistema, rang = {rank_A}")

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Gauss metodi"}
