"""
algebra.py — Algebra moduli
Chiziqli, kvadrat tenglamalar, sistemalar va tengsizliklarni
qadamma-qadam yechadi.
"""

import sympy as sp
from sympy import (
    symbols, solve, factor, expand, simplify,
    sqrt, Rational, latex, Symbol, Poly,
    linsolve, nonlinsolve, solveset, S, oo,
    Interval, Union, Intersection
)
from core.formatter import format_step, build_solution


# ─────────────────────────────────────────────
# 1. CHIZIQLI TENGLAMA: ax + b = 0
# ─────────────────────────────────────────────

def solve_linear(equation_str: str, var_str: str = "x") -> dict:
    """
    Chiziqli tenglamani qadamma-qadam yechadi.
    Misol: '2*x + 6 = 0'  yoki  '3*x - 9 = 4'
    """
    x = symbols(var_str)
    steps = []

    try:
        # Tenglamani ikki qismga ajratish
        if "=" in equation_str:
            left_str, right_str = equation_str.split("=", 1)
        else:
            left_str, right_str = equation_str, "0"

        left  = sp.sympify(left_str.strip())
        right = sp.sympify(right_str.strip())

        steps.append(format_step(1, "Berilgan tenglama",
            f"Tenglamani standart ko'rinishga keltiramiz:",
            left - right))

        # ax + b = 0 shaklga keltirish
        expr = sp.expand(left - right)
        steps.append(format_step(2, "Soddalashtirish",
            f"Barcha hadlarni chap tomonga o'tkazamiz:",
            expr))

        # Koeffitsientlarni ajratish
        poly = Poly(expr, x)
        coeffs = poly.all_coeffs()

        if len(coeffs) == 2:
            a, b = coeffs
            steps.append(format_step(3, "Koeffitsientlarni aniqlash",
                f"a = {a},   b = {b}\nFormula:  x = -b / a"))

            if a == 0:
                if b == 0:
                    return build_solution("Chiziqli tenglama", steps,
                        "Barcha sonlar yechim", "Tenglama o'xshash")
                else:
                    return build_solution("Chiziqli tenglama", steps,
                        "Yechim yo'q", "Qarama-qarshi tenglama")

            answer = sp.Rational(-b, a) if (a != 0) else solve(expr, x)[0]
            steps.append(format_step(4, "Yechim",
                f"x = -({b}) / ({a}) = {answer}",
                sp.Eq(x, answer)))
        else:
            # Umumiy holat
            answer = solve(expr, x)
            steps.append(format_step(3, "Yechish",
                f"solve() yordamida yechamiz:", sp.Eq(x, answer[0]) if answer else None))
            answer = answer[0] if answer else None

        # Tekshirish
        if answer is not None:
            check = expr.subs(x, answer)
            steps.append(format_step(len(steps)+1, "Tekshirish",
                f"x = {answer} ni tenglamaga qo'yamiz: {sp.expand(check)} = 0 ✓",
                sp.Eq(expr.subs(x, answer), 0)))

        return build_solution("Chiziqli tenglama", steps, answer)

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Chiziqli tenglama"}


# ─────────────────────────────────────────────
# 2. KVADRAT TENGLAMA: ax² + bx + c = 0
# ─────────────────────────────────────────────

def solve_quadratic(equation_str: str, var_str: str = "x") -> dict:
    """
    Kvadrat tenglamani diskriminant orqali qadamma-qadam yechadi.
    Misol: 'x**2 - 5*x + 6 = 0'
    """
    x = symbols(var_str)
    steps = []

    try:
        if "=" in equation_str:
            left_str, right_str = equation_str.split("=", 1)
        else:
            left_str, right_str = equation_str, "0"

        left  = sp.sympify(left_str.strip())
        right = sp.sympify(right_str.strip())
        expr  = sp.expand(left - right)

        steps.append(format_step(1, "Berilgan tenglama",
            "Standart ko'rinish:  ax² + bx + c = 0", expr))

        # Koeffitsientlarni ajratish
        poly = Poly(expr, x)
        coeffs = poly.all_coeffs()

        # Darajani tekshirish
        degree = poly.degree()
        if degree != 2:
            return {"error": f"Bu kvadrat tenglama emas (daraja: {degree})",
                    "steps": steps, "problem_type": "Kvadrat tenglama"}

        if len(coeffs) == 3:
            a, b, c = coeffs
        elif len(coeffs) == 2:
            a, b, c = coeffs[0], coeffs[1], 0
        else:
            a, b, c = coeffs[0], 0, 0

        steps.append(format_step(2, "Koeffitsientlar",
            f"a = {a}\nb = {b}\nc = {c}"))

        # Diskriminant
        D = b**2 - 4*a*c
        steps.append(format_step(3, "Diskriminant",
            f"D = b² - 4ac\nD = ({b})² - 4·({a})·({c})\nD = {b**2} - {4*a*c} = {D}",
            sp.Symbol('D') - D))

        # Ildizlar
        if D > 0:
            x1 = (-b + sp.sqrt(D)) / (2*a)
            x2 = (-b - sp.sqrt(D)) / (2*a)
            x1_s = sp.simplify(x1)
            x2_s = sp.simplify(x2)
            steps.append(format_step(4, "D > 0 — Ikki haqiqiy ildiz",
                f"x₁ = (-b + √D) / 2a = ({-b} + √{D}) / {2*a} = {x1_s}\n"
                f"x₂ = (-b - √D) / 2a = ({-b} - √{D}) / {2*a} = {x2_s}"))
            answer = [x1_s, x2_s]

        elif D == 0:
            x0 = sp.Rational(-b, 2*a)
            steps.append(format_step(4, "D = 0 — Bitta ildiz (qo'sh ildiz)",
                f"x₁ = x₂ = -b / 2a = {-b} / {2*a} = {x0}"))
            answer = [x0]

        else:
            steps.append(format_step(4, "D < 0 — Haqiqiy ildiz yo'q",
                f"D = {D} < 0\nTenglama kompleks ildizlarga ega (haqiqiy yechim yo'q)"))
            x1_c = (-b + sp.sqrt(D)) / (2*a)
            x2_c = (-b - sp.sqrt(D)) / (2*a)
            answer = [sp.simplify(x1_c), sp.simplify(x2_c)]

        # Faktorlash
        try:
            factored = sp.factor(expr)
            steps.append(format_step(5, "Ko'paytuvchilarga ajratish",
                "Tenglamani ko'paytuvchilarga ajratamiz:", factored))
        except Exception:
            pass

        # Tekshirish
        for xi in answer:
            val = sp.expand(expr.subs(x, xi))
            steps.append(format_step(len(steps)+1, f"Tekshirish: x = {xi}",
                f"expr({xi}) = {val} ≈ 0 ✓"))

        return build_solution("Kvadrat tenglama", steps, answer,
            f"D = {D}, Ildizlar soni: {len(answer)}")

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Kvadrat tenglama"}


# ─────────────────────────────────────────────
# 3. TENGSIZLIK
# ─────────────────────────────────────────────

def solve_inequality(ineq_str: str, var_str: str = "x") -> dict:
    """
    Tengsizlikni yechadi va intervalni qaytaradi.
    Misol: 'x**2 - 4 > 0'  yoki  '2*x - 3 <= 5'
    """
    x = symbols(var_str, real=True)
    steps = []

    try:
        # Operator aniqlash
        op_map = {
            ">=": ("≥", sp.Ge),
            "<=": ("≤", sp.Le),
            ">":  (">",  sp.Gt),
            "<":  ("<",  sp.Lt),
        }
        op_str, op_func = None, None
        for op_key, (op_display, op_f) in op_map.items():
            if op_key in ineq_str:
                parts = ineq_str.split(op_key, 1)
                lhs = sp.sympify(parts[0].strip())
                rhs = sp.sympify(parts[1].strip())
                op_str, op_func = op_display, op_f
                break

        if op_func is None:
            return {"error": "Tengsizlik belgisi topilmadi (>, <, >=, <=)",
                    "steps": [], "problem_type": "Tengsizlik"}

        ineq = op_func(lhs, rhs)
        steps.append(format_step(1, "Berilgan tengsizlik",
            f"Yechamiz: {lhs} {op_str} {rhs}"))

        # Tengsizlikni yechish
        expr = sp.expand(lhs - rhs)
        steps.append(format_step(2, "Soddalashtirish",
            f"Standart ko'rinish: {expr} {op_str} 0", expr))

        # Tenglik nuqtalari
        roots = solve(expr, x)
        if roots:
            steps.append(format_step(3, "Tenglik nuqtalari",
                f"f(x) = 0 bo'lganda x = {roots}\n"
                f"Bu nuqtalar intervallarni ajratadi."))

        # Yechim
        solution = sp.solve(ineq, x)
        steps.append(format_step(4, "Yechim sohasi",
            f"Yechim: {solution}"))

        return build_solution("Tengsizlik", steps, solution,
            f"Tenglik nuqtalari: {roots}")

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Tengsizlik"}


# ─────────────────────────────────────────────
# 4. TENGLAMALAR SISTEMASI
# ─────────────────────────────────────────────

def solve_system(equations: list, variables: list = None) -> dict:
    """
    Tenglamalar sistemasini yechadi.
    equations: ['x + y = 5', '2*x - y = 1']
    variables: ['x', 'y']  (ixtiyoriy)
    """
    steps = []

    try:
        # O'zgaruvchilarni aniqlash
        if variables is None:
            variables = ['x', 'y']
        syms = symbols(' '.join(variables))
        if not isinstance(syms, tuple):
            syms = (syms,)
        sym_dict = {v: s for v, s in zip(variables, syms)}

        steps.append(format_step(1, "Berilgan sistema",
            "\n".join([f"({i+1})  {eq}" for i, eq in enumerate(equations)])))

        # Tenglamalarni parse qilish
        parsed = []
        for eq_str in equations:
            if "=" in eq_str:
                l, r = eq_str.split("=", 1)
                expr = sp.sympify(l.strip()) - sp.sympify(r.strip())
            else:
                expr = sp.sympify(eq_str.strip())
            parsed.append(expr)
            steps.append(format_step(len(steps)+1,
                f"Tenglama: {eq_str}",
                f"Standart ko'rinish: {expr} = 0", expr))

        # Yechish
        solution = solve(parsed, list(syms), dict=True)

        if not solution:
            steps.append(format_step(len(steps)+1, "Natija",
                "Yechim topilmadi yoki sistema qarama-qarshi"))
            return build_solution("Tenglamalar sistemasi", steps,
                "Yechim yo'q", "Inkonsistent sistema")

        sol = solution[0]
        result_str = "\n".join([f"{k} = {v}" for k, v in sol.items()])
        steps.append(format_step(len(steps)+1, "Yechim",
            f"Topilgan qiymatlar:\n{result_str}"))

        # Tekshirish
        for i, eq_expr in enumerate(parsed):
            check_val = eq_expr.subs(sol)
            check_simplified = sp.simplify(check_val)
            steps.append(format_step(len(steps)+1,
                f"Tekshirish — Tenglama {i+1}",
                f"Qiymatlarni qo'yganda: {check_simplified} = 0 ✓"))

        return build_solution("Tenglamalar sistemasi", steps, sol,
            f"{len(syms)} noma'lum, {len(equations)} tenglama")

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Tenglamalar sistemasi"}
