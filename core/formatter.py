"""
formatter.py — Qadamma-qadam tushuntirish va LaTeX formatlash moduli
"""
import sympy as sp
from sympy import latex as sp_latex
import re, io


def _expr_to_latex(text: str) -> str | None:
    """
    Oddiy matematik matndagi ifodalarni LaTeX ga o'tkazishga harakat qiladi.
    Muvaffaqiyatsiz bo'lsa None qaytaradi.
    """
    try:
        proc = re.sub(r'\^', '**', text.strip())
        proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', proc)
        _ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
        expr = sp.sympify(proc, locals=_ns)
        return sp_latex(expr)
    except Exception:
        return None


def format_step(step_number: int, title: str, content: str,
                latex_expr=None) -> dict:
    """
    Bitta qadamni formatlaydi.
    latex_expr: SymPy ob'ekt — to'g'ridan-to'g'ri LaTeX sifatida ko'rsatiladi.
    content: inson o'qiydigan izoh matni.
    """
    ltx = None
    if latex_expr is not None:
        try:
            ltx = sp_latex(latex_expr)
        except Exception:
            ltx = str(latex_expr)

    return {
        "step":    step_number,
        "title":   title,
        "content": content,
        "latex":   ltx,
    }


def build_solution(problem_type: str, steps: list, answer,
                   extra_info: str = "") -> dict:
    try:
        ans_ltx = sp_latex(answer)
    except Exception:
        ans_ltx = str(answer)

    return {
        "problem_type": problem_type,
        "steps":        steps,
        "answer":       answer,
        "answer_latex": ans_ltx,
        "extra_info":   extra_info,
    }
