"""
numerical.py — Sonli metodlar moduli
Nyuton, biseksiya, oddiy iteratsiya metodlari.
"""

import sympy as sp
from sympy import symbols, diff, lambdify, latex
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from core.formatter import format_step, build_solution


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# 1. NYUTON METODI (Newton-Raphson)
# ─────────────────────────────────────────────

def newton_method(func_str: str, x0: float, tol: float = 1e-7,
                  max_iter: int = 50, var_str: str = "x") -> dict:
    """
    Nyuton-Rafson iteratsiya metodi.
    f(x) = 0 tenglamani x0 boshlang'ich qiymatidan iteratsiya bilan yechadi.

    Formula: x_{n+1} = x_n - f(x_n)/f'(x_n)
    """
    var = symbols(var_str)
    steps = []

    try:
        f_sym  = sp.sympify(func_str)
        fp_sym = diff(f_sym, var)

        steps.append(format_step(1, "Berilgan funksiya va boshlang'ich qiymat",
            f"f(x) = {func_str}\n"
            f"f'(x) = {fp_sym}\n"
            f"x₀ = {x0},  aniqlik = {tol}"))

        steps.append(format_step(2, "Nyuton-Rafson formulasi",
            "x_{n+1} = x_n - f(x_n) / f'(x_n)\n\n"
            "Bu formula funksiyaning tangent chizig'i orqali\n"
            "ildizga yaqinlashish tamoyiliga asoslanadi."))

        f_num  = lambdify(var, f_sym,  'numpy')
        fp_num = lambdify(var, fp_sym, 'numpy')

        x_cur = float(x0)
        iterations = []
        converged = False

        for i in range(max_iter):
            fx  = f_num(x_cur)
            fpx = fp_num(x_cur)

            if abs(fpx) < 1e-15:
                steps.append(format_step(3, f"Iteratsiya {i+1}",
                    f"f'(x) ≈ 0 — metod to'xtadi (nol hosilaga bo'linish)"))
                break

            x_new  = x_cur - fx / fpx
            error  = abs(x_new - x_cur)
            iterations.append((i+1, x_cur, fx, fpx, x_new, error))

            if i < 6:  # Faqat dastlabki 6 qadamni ko'rsatish
                steps.append(format_step(3, f"Iteratsiya {i+1}",
                    f"x_{i} = {x_cur:.8f}\n"
                    f"f(x_{i}) = {fx:.2e}\n"
                    f"f'(x_{i}) = {fpx:.6f}\n"
                    f"x_{i+1} = {x_cur:.6f} - ({fx:.2e})/({fpx:.6f})\n"
                    f"x_{i+1} = {x_new:.8f}\n"
                    f"|x_{i+1} - x_{i}| = {error:.2e}"))
            elif i == 6:
                steps.append(format_step(3, "... (qolgan iteratsiyalar)",
                    "Yaqinlashish davom etmoqda..."))

            if error < tol:
                converged = True
                steps.append(format_step(4, f"Yaqinlashish — {i+1} ta iteratsiyadan so'ng",
                    f"|x_{i+1} - x_{i}| = {error:.2e} < {tol}\n"
                    f"Yechim: x ≈ {x_new:.10f}\n"
                    f"f(x) = {f_num(x_new):.2e} ≈ 0 ✓"))
                x_cur = x_new
                break
            x_cur = x_new

        if not converged:
            steps.append(format_step(4, "Ogohlantirish",
                f"Metod {max_iter} iteratsiyada yaqinlashmadi.\n"
                f"Oxirgi qiymat: x ≈ {x_cur:.8f}"))

        # Grafik
        fig = _plot_newton(f_sym, fp_sym, var, iterations[:8], float(x0))
        img = _fig_to_base64(fig) if fig else None

        sol = build_solution("Nyuton metodi", steps, x_cur,
            f"Iteratsiyalar soni: {len(iterations)}, Yaqinlashdi: {converged}")
        sol["plot"] = img
        sol["table"] = iterations
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Nyuton metodi"}


def _plot_newton(f_sym, fp_sym, var, iters, x0):
    """Nyuton metodini vizualizatsiya qiladi."""
    try:
        f_num  = lambdify(var, f_sym,  'numpy')
        fp_num = lambdify(var, fp_sym, 'numpy')

        all_x = [x0] + [it[4] for it in iters]
        xmin  = min(all_x) - 2
        xmax  = max(all_x) + 2
        x_arr = np.linspace(xmin, xmax, 600)
        y_arr = f_num(x_arr)
        y_arr = np.where(np.abs(y_arr) > 30, np.nan, y_arr)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_arr, y_arr, 'royalblue', lw=2.5, label=f'f(x) = {f_sym}')
        ax.axhline(0, color='black', lw=1)

        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(iters)))
        for i, (step, xn, fxn, fpxn, xn1, err) in enumerate(iters):
            ax.plot([xn, xn1], [fxn, 0], '--', color=colors[i], lw=1.5, alpha=0.7)
            ax.plot(xn, fxn, 'o', color=colors[i], markersize=6)
            ax.plot(xn1, 0, 's', color=colors[i], markersize=5)
            ax.annotate(f'x{step}={xn1:.3f}', (xn1, 0),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=8, color='darkred', ha='center')

        if iters:
            ax.plot(iters[-1][4], 0, '*', color='red', markersize=14,
                    zorder=10, label=f'Yechim ≈ {iters[-1][4]:.6f}')

        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.set_title("Nyuton-Rafson metodi — iteratsiyalar", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────
# 2. BISEKSIYA METODI (Ikiga bo'lish)
# ─────────────────────────────────────────────

def bisection_method(func_str: str, a: float, b: float,
                     tol: float = 1e-6, max_iter: int = 60) -> dict:
    """
    Biseksiya (ikiga bo'lish) metodi.
    [a, b] oraliqda f(a)*f(b) < 0 shart bajarilishi kerak.
    """
    x = symbols('x')
    steps = []

    try:
        f_sym = sp.sympify(func_str)
        f_num = lambdify(x, f_sym, 'numpy')

        fa, fb = f_num(float(a)), f_num(float(b))
        steps.append(format_step(1, "Boshlang'ich ma'lumotlar",
            f"f(x) = {func_str}\n"
            f"[a, b] = [{a}, {b}]\n"
            f"f({a}) = {fa:.6f},  f({b}) = {fb:.6f}"))

        if fa * fb > 0:
            return {"error": f"f({a})·f({b}) > 0: Oraliqda ildiz mavjud emas yoki juft sonli ildizlar bor",
                    "steps": steps, "problem_type": "Biseksiya"}

        steps.append(format_step(2, "Mavjudlik sharti",
            f"f({a})·f({b}) = {fa*fb:.4f} < 0 ✓\n"
            f"Oraliqda hech bo'lmaganda bitta ildiz mavjud (Bolzano teoremasi)"))

        steps.append(format_step(3, "Biseksiya formulasi",
            "c = (a + b) / 2\n"
            "• f(a)·f(c) < 0 → yangi b = c\n"
            "• f(b)·f(c) < 0 → yangi a = c"))

        a_cur, b_cur = float(a), float(b)
        iters, c_cur = [], None
        converged = False

        for i in range(max_iter):
            c_cur = (a_cur + b_cur) / 2
            fc    = f_num(c_cur)
            err   = (b_cur - a_cur) / 2
            iters.append((i+1, a_cur, b_cur, c_cur, fc, err))

            if i < 8:
                steps.append(format_step(3, f"Iteratsiya {i+1}",
                    f"a = {a_cur:.6f},  b = {b_cur:.6f}\n"
                    f"c = (a+b)/2 = {c_cur:.8f}\n"
                    f"f(c) = {fc:.2e}\n"
                    f"Yangi interval uzunligi: {err:.2e}"))
            elif i == 8:
                steps.append(format_step(3, "...", "Iteratsiyalar davom etmoqda..."))

            if abs(fc) < tol or err < tol:
                converged = True
                steps.append(format_step(4, f"Yaqinlashish — {i+1}-iteratsiya",
                    f"Aniqlik: {err:.2e} < {tol}\n"
                    f"Yechim: x ≈ {c_cur:.10f}\n"
                    f"f(x) = {fc:.2e} ≈ 0 ✓"))
                break

            if f_num(a_cur) * fc < 0:
                b_cur = c_cur
            else:
                a_cur = c_cur

        if not converged:
            steps.append(format_step(4, "Natija",
                f"Maksimal iteratsiyalarga yetildi.\nYechim ≈ {c_cur:.8f}"))

        # Grafik
        fig = _plot_bisection(f_sym, x, iters[:10], float(a), float(b))
        img = _fig_to_base64(fig) if fig else None

        sol = build_solution("Biseksiya metodi", steps, c_cur,
            f"Iteratsiyalar: {len(iters)}, Yaqinlashdi: {converged}")
        sol["plot"] = img
        sol["table"] = iters
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Biseksiya"}


def _plot_bisection(f_sym, var, iters, a, b):
    """Biseksiya metodini vizualizatsiya qiladi."""
    try:
        f_num = lambdify(var, f_sym, 'numpy')
        x_arr = np.linspace(a - 0.5, b + 0.5, 500)
        y_arr = f_num(x_arr)
        y_arr = np.where(np.abs(y_arr) > 20, np.nan, y_arr)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_arr, y_arr, 'royalblue', lw=2.5, label=f'f(x) = {f_sym}')
        ax.axhline(0, color='black', lw=1)

        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(iters)))
        for i, (step, ai, bi, ci, fc, err) in enumerate(iters):
            ax.axvspan(ai, bi, alpha=0.06, color=colors[i])
            ax.plot(ci, fc, 'o', color=colors[i], markersize=5)

        if iters:
            final_c = iters[-1][3]
            ax.axvline(final_c, color='red', linestyle='--', lw=1.5, alpha=0.8,
                       label=f'Yechim ≈ {final_c:.6f}')
            ax.plot(final_c, 0, '*', color='red', markersize=14, zorder=10)

        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.set_title("Biseksiya metodi", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. ODDIY ITERATSIYA METODI
# ─────────────────────────────────────────────

def fixed_point_iteration(phi_str: str, x0: float, tol: float = 1e-7,
                           max_iter: int = 100) -> dict:
    """
    Oddiy iteratsiya metodi: x = φ(x)
    x_{n+1} = φ(x_n)
    """
    x = symbols('x')
    steps = []

    try:
        phi_sym = sp.sympify(phi_str)
        phi_num = lambdify(x, phi_sym, 'numpy')

        steps.append(format_step(1, "Iteratsiya funksiyasi",
            f"φ(x) = {phi_str}\n"
            f"x₀ = {x0},  aniqlik = {tol}\n\n"
            f"x_{{n+1}} = φ(x_n) iteratsiyasini amalga oshiramiz"))

        # Yaqinlashish sharti
        phi_prime = diff(phi_sym, x)
        steps.append(format_step(2, "Yaqinlashish sharti",
            f"φ'(x) = {phi_prime}\n"
            f"|φ'(x)| < 1 bo'lsa metod yaqinlashadi"))

        x_cur = float(x0)
        iters = []
        converged = False

        for i in range(max_iter):
            try:
                x_new = float(phi_num(x_cur))
            except Exception:
                steps.append(format_step(3, "Xato", "Iteratsiyada son xatosi"))
                break

            err = abs(x_new - x_cur)
            iters.append((i+1, x_cur, x_new, err))

            if i < 8:
                steps.append(format_step(3, f"Iteratsiya {i+1}",
                    f"x_{i} = {x_cur:.8f}\n"
                    f"x_{i+1} = φ({x_cur:.6f}) = {x_new:.8f}\n"
                    f"|x_{i+1} - x_i| = {err:.2e}"))
            elif i == 8:
                steps.append(format_step(3, "...", "Iteratsiyalar davom etmoqda..."))

            if err < tol:
                converged = True
                steps.append(format_step(4, f"Yaqinlashish — {i+1}-iteratsiya",
                    f"Yechim: x ≈ {x_new:.10f}\n"
                    f"Aniqlik: {err:.2e} ✓"))
                x_cur = x_new
                break
            x_cur = x_new

        sol = build_solution("Oddiy iteratsiya", steps, x_cur,
            f"Iteratsiyalar: {len(iters)}, Yaqinlashdi: {converged}")
        sol["table"] = iters
        return sol

    except Exception as e:
        return {"error": str(e), "steps": [], "problem_type": "Oddiy iteratsiya"}
