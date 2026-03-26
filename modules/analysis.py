"""
analysis.py — Matematik analiz moduli
Limitlar, hosilalar, integrallar va grafik chizish.
"""

import sympy as sp
from sympy import (
    symbols, limit, diff, integrate, oo, latex,
    simplify, expand, factor, series, Symbol,
    sin, cos, tan, exp, log, sqrt, pi, E,
    Piecewise, Abs
)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from core.formatter import format_step, build_solution


x, t, n = symbols('x t n')


def _fig_to_base64(fig) -> str:
    """Matplotlib figurani base64 stringga o'tkazadi (Streamlit uchun)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


# ─────────────────────────────────────────────
# 1. LIMIT
# ─────────────────────────────────────────────

def compute_limit(func_str: str, point_str: str, direction: str = "+-",
                  var_str: str = "x") -> dict:
    """
    Limitni hisoblaydi.
    Misol: func_str='sin(x)/x', point_str='0'
    direction: '+', '-', '+-' (ikki tomonlama)
    """
    var = symbols(var_str)
    steps = []

    try:
        f = sp.sympify(func_str)
        point = sp.sympify(point_str.replace("oo", "oo").replace("inf", "oo"))

        steps.append(format_step(1, "Berilgan limit",
            f"lim[{var_str}→{point_str}]  {func_str}", f))

        # Bevosita qo'yib ko'rish
        try:
            direct_val = f.subs(var, point)
            direct_simplified = sp.simplify(direct_val)
            if direct_simplified.is_number and not direct_simplified.has(sp.zoo, sp.nan):
                steps.append(format_step(2, "Bevosita qo'yish",
                    f"f({point_str}) = {direct_simplified}\nBu aniq qiymat, limit = {direct_simplified}"))
        except Exception:
            steps.append(format_step(2, "Bevosita qo'yish",
                f"f({point_str}) da noaniqlik mavjud — qo'shimcha usul kerak"))

        # Haqiqiy limit hisoblash
        if direction == "+-":
            result = limit(f, var, point)
            lim_right = limit(f, var, point, '+')
            lim_left  = limit(f, var, point, '-')

            steps.append(format_step(3, "O'ng tomondan limit (x→a⁺)",
                f"lim[x→{point}⁺] = {lim_right}"))
            steps.append(format_step(4, "Chap tomondan limit (x→a⁻)",
                f"lim[x→{point}⁻] = {lim_left}"))

            if lim_right == lim_left:
                steps.append(format_step(5, "Xulosa",
                    f"Chap = O'ng = {result}\nLimit mavjud: lim = {result}", result))
            else:
                steps.append(format_step(5, "Xulosa",
                    f"Chap ({lim_left}) ≠ O'ng ({lim_right})\nLimit mavjud EMAS!"))
                result = sp.nan

        elif direction == "+":
            result = limit(f, var, point, '+')
            steps.append(format_step(3, "O'ng tomondan limit",
                f"lim[x→{point}⁺] = {result}"))
        else:
            result = limit(f, var, point, '-')
            steps.append(format_step(3, "Chap tomondan limit",
                f"lim[x→{point}⁻] = {result}"))

        # Grafik
        fig = _plot_limit(f, var, point, result)
        img = _fig_to_base64(fig) if fig else None

        sol = build_solution("Limit", steps, result)
        sol["plot"] = img
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Limit"}


def _plot_limit(f, var, point, result):
    """Limit grafigini chizadi."""
    try:
        point_float = float(point)
    except Exception:
        point_float = 0

    x_range = np.linspace(point_float - 3, point_float + 3, 500)
    x_range = x_range[np.abs(x_range - point_float) > 1e-6]

    f_lam = sp.lambdify(var, f, modules=['numpy'])
    try:
        y_vals = f_lam(x_range)
        y_vals = np.where(np.abs(y_vals) > 50, np.nan, y_vals)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_range, y_vals, 'royalblue', linewidth=2, label=f'f(x) = {f}')
    try:
        res_float = float(result)
        ax.axhline(res_float, color='tomato', linestyle='--', alpha=0.6,
                   label=f'Limit = {result}')
        ax.plot(point_float, res_float, 'ro', markersize=8, zorder=5, label=f'x = {point}')
    except Exception:
        pass
    ax.axvline(point_float, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'lim f(x)  as  x → {point}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 2. HOSILA (DIFFERENSIALLASH)
# ─────────────────────────────────────────────

def compute_derivative(func_str: str, var_str: str = "x", order: int = 1) -> dict:
    """
    Hosilani hisoblaydi (n-tartibli).
    Misol: func_str='x**3 + 2*x', order=1
    """
    var = symbols(var_str)
    steps = []

    try:
        f = sp.sympify(func_str)

        steps.append(format_step(1, "Berilgan funksiya",
            f"f({var_str}) = {func_str}", f))

        # Differensiallash qoidalarini tushuntirish
        steps.append(format_step(2, "Differensiallash qoidalari",
            "• (xⁿ)' = n·xⁿ⁻¹\n• (u±v)' = u' ± v'\n"
            "• (uv)' = u'v + uv'\n• (sin x)' = cos x\n"
            "• (cos x)' = -sin x\n• (eˣ)' = eˣ\n• (ln x)' = 1/x"))

        # Har bir darajali had uchun tushuntirish
        if f.is_Add:
            terms = f.as_ordered_terms()
            steps.append(format_step(3, "Hadma-had differensiallash",
                f"f(x) = {' + '.join([str(t) for t in terms])}"))
            for i, term in enumerate(terms):
                term_deriv = diff(term, var)
                steps.append(format_step(3, f"  ({i+1}-had) d/dx[{term}]",
                    f"= {term_deriv}", term_deriv))

        # Asosiy hosila
        result = diff(f, var, order)
        result_simplified = sp.simplify(result)

        order_str = {1: "birinchi", 2: "ikkinchi", 3: "uchinchi"}.get(order, f"{order}-tartibli")
        steps.append(format_step(4, f"f'(x) — {order_str} tartibli hosila",
            f"d^{order}/d{var_str}^{order} [{func_str}] = {result_simplified}",
            result_simplified))

        # Kritik nuqtalar (faqat 1-tartibli hosila uchun)
        if order == 1:
            critical = sp.solve(result_simplified, var)
            if critical:
                steps.append(format_step(5, "Kritik nuqtalar (f'(x) = 0)",
                    f"{var_str} = {critical}"))
                # Extremum tekshirish
                f2 = diff(f, var, 2)
                for cp in critical:
                    f2_val = f2.subs(var, cp)
                    if f2_val > 0:
                        ftype = "Mahalliy minimum"
                    elif f2_val < 0:
                        ftype = "Mahalliy maksimum"
                    else:
                        ftype = "Burilish nuqtasi yoki tekshirish kerak"
                    fval = f.subs(var, cp)
                    steps.append(format_step(5,
                        f"  x = {cp}: {ftype}",
                        f"f''({cp}) = {f2_val},  f({cp}) = {fval}"))

        # Grafik
        fig = _plot_derivative(f, result_simplified, var)
        img = _fig_to_base64(fig) if fig else None

        sol = build_solution("Hosila", steps, result_simplified,
            f"{order_str} tartibli hosila")
        sol["plot"] = img
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Hosila"}


def _plot_derivative(f, f_prime, var):
    """Funksiya va uning hosilasini chizadi."""
    try:
        x_vals = np.linspace(-5, 5, 500)
        f_lam  = sp.lambdify(var, f,       modules=['numpy'])
        fp_lam = sp.lambdify(var, f_prime, modules=['numpy'])
        y_f    = np.array(f_lam(x_vals), dtype=float)
        y_fp   = np.array(fp_lam(x_vals), dtype=float)
        y_f    = np.where(np.abs(y_f)  > 30, np.nan, y_f)
        y_fp   = np.where(np.abs(y_fp) > 30, np.nan, y_fp)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_f,  'royalblue', lw=2, label=f'f(x) = {f}')
        ax.plot(x_vals, y_fp, 'tomato',    lw=2, linestyle='--', label=f"f'(x) = {f_prime}")
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title("Funksiya va uning hosilasi", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. INTEGRAL
# ─────────────────────────────────────────────

def compute_integral(func_str: str, var_str: str = "x",
                     lower=None, upper=None) -> dict:
    """
    Aniq yoki noaniq integralni hisoblaydi.
    lower, upper: aniq integral uchun chegara ('0', '1', 'pi' ...)
    """
    var = symbols(var_str)
    steps = []

    try:
        f = sp.sympify(func_str)
        is_definite = (lower is not None and upper is not None)

        if is_definite:
            a = sp.sympify(str(lower).replace("pi", "pi").replace("oo", "oo"))
            b = sp.sympify(str(upper).replace("pi", "pi").replace("oo", "oo"))
            steps.append(format_step(1, "Berilgan aniq integral",
                f"∫[{lower} → {upper}]  {func_str}  d{var_str}", f))
        else:
            steps.append(format_step(1, "Berilgan noaniq integral",
                f"∫ {func_str} d{var_str}", f))

        # Integrallash qoidalari
        steps.append(format_step(2, "Integrallash qoidalari",
            "• ∫xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
            "• ∫sin(x) dx = -cos(x) + C\n"
            "• ∫cos(x) dx = sin(x) + C\n"
            "• ∫eˣ dx = eˣ + C\n"
            "• ∫1/x dx = ln|x| + C"))

        # Noaniq integral
        indef = integrate(f, var)
        indef_simplified = sp.simplify(indef)
        steps.append(format_step(3, "Noaniq integral (antiturunma)",
            f"F(x) = ∫f(x)dx = {indef_simplified} + C",
            indef_simplified))

        # Tekshirish: F'(x) = f(x) ?
        check = sp.simplify(diff(indef_simplified, var) - f)
        steps.append(format_step(4, "Tekshirish: F'(x) = f(x) ?",
            f"F'(x) - f(x) = {check} → {'✓ To\'g\'ri' if check == 0 else '⚠ Farq bor'}"))

        if is_definite:
            # Nyuton-Leybnits formulasi
            steps.append(format_step(5, "Nyuton-Leybnits formulasi",
                f"∫[a→b] f(x)dx = F(b) - F(a)"))

            Fb = indef_simplified.subs(var, b)
            Fa = indef_simplified.subs(var, a)
            Fb_s = sp.simplify(Fb)
            Fa_s = sp.simplify(Fa)
            steps.append(format_step(6, f"F(b) = F({upper})",
                f"F({upper}) = {Fb_s}"))
            steps.append(format_step(7, f"F(a) = F({lower})",
                f"F({lower}) = {Fa_s}"))

            result = sp.simplify(Fb_s - Fa_s)
            steps.append(format_step(8, "Aniq integral qiymati",
                f"F({upper}) - F({lower}) = {Fb_s} - ({Fa_s}) = {result}",
                result))

            # Grafik
            fig = _plot_integral(f, var, a, b)
        else:
            result = indef_simplified
            fig = _plot_function(f, var, func_str)

        img = _fig_to_base64(fig) if fig else None
        sol = build_solution(
            "Aniq integral" if is_definite else "Noaniq integral",
            steps, result,
            f"Maydon ≈ {result}" if is_definite else "Antiturunma (C qo'shiladi)")
        sol["plot"] = img
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Integral"}


def _plot_integral(f, var, a, b):
    """Aniq integral maydonini chizadi."""
    try:
        af, bf = float(a), float(b)
        x_all  = np.linspace(af - 1, bf + 1, 500)
        x_fill = np.linspace(af, bf, 300)
        f_lam  = sp.lambdify(var, f, modules=['numpy'])
        y_all  = np.array(f_lam(x_all),  dtype=float)
        y_fill = np.array(f_lam(x_fill), dtype=float)
        y_all  = np.where(np.abs(y_all)  > 30, np.nan, y_all)
        y_fill = np.where(np.abs(y_fill) > 30, np.nan, y_fill)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_all, y_all, 'royalblue', lw=2.5, label=f'f(x) = {f}')
        ax.fill_between(x_fill, y_fill, alpha=0.3, color='royalblue',
                        label=f'Maydon [{af}, {bf}]')
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Aniq integral: ∫[{af}, {bf}] f(x) dx', fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def _plot_function(f, var, label=""):
    """Oddiy funksiya grafigi."""
    try:
        x_vals = np.linspace(-5, 5, 500)
        f_lam  = sp.lambdify(var, f, modules=['numpy'])
        y_vals = np.array(f_lam(x_vals), dtype=float)
        y_vals = np.where(np.abs(y_vals) > 30, np.nan, y_vals)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_vals, 'royalblue', lw=2.5, label=f'f(x) = {label}')
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title("Funksiya grafigi", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────
# 4. GRAFIK CHIZISH (mustaqil)
# ─────────────────────────────────────────────

def plot_function(func_str: str, x_min: float = -10, x_max: float = 10,
                  var_str: str = "x") -> dict:
    """
    Funksiya grafigini chizadi.
    """
    var = symbols(var_str)
    steps = []
    try:
        f = sp.sympify(func_str)
        steps.append(format_step(1, "Berilgan funksiya",
            f"f(x) = {func_str}", f))

        # Muhim nuqtalar
        f_prime = diff(f, var)
        critical = sp.solve(f_prime, var)
        if critical:
            steps.append(format_step(2, "Kritik nuqtalar",
                f"f'(x) = {f_prime}\nf'(x) = 0  ⟹  x = {critical}"))

        x_vals = np.linspace(x_min, x_max, 800)
        f_lam  = sp.lambdify(var, f, modules=['numpy'])
        try:
            y_vals = np.array(f_lam(x_vals), dtype=float)
        except Exception:
            y_vals = np.full_like(x_vals, np.nan)
        y_vals = np.where(np.abs(y_vals) > 1e6, np.nan, y_vals)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, y_vals, 'royalblue', lw=2.5, label=f'f(x) = {func_str}')

        # Kritik nuqtalarni belgilash
        for cp in (critical or []):
            try:
                cp_f = float(cp)
                fcp  = float(f.subs(var, cp))
                ax.plot(cp_f, fcp, 'ro', markersize=7, zorder=5)
                ax.annotate(f'({cp_f:.2f}, {fcp:.2f})',
                            (cp_f, fcp), textcoords="offset points",
                            xytext=(8, 8), fontsize=9, color='darkred')
            except Exception:
                pass

        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'f(x) = {func_str}', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        fig.tight_layout()

        img = _fig_to_base64(fig)
        sol = build_solution("Grafik", steps, func_str)
        sol["plot"] = img
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Grafik"}
