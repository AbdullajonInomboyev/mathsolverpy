"""
server.py — MathSolverPy Flask API Server
Barcha matematik hisob-kitoblar uchun REST API.

Ishga tushirish:
    python server.py

API routes:
    POST /api/algebra/linear
    POST /api/algebra/quadratic
    POST /api/algebra/inequality
    POST /api/algebra/system
    POST /api/analysis/limit
    POST /api/analysis/derivative
    POST /api/analysis/integral
    POST /api/analysis/plot
    POST /api/numerical/newton
    POST /api/numerical/bisection
    POST /api/numerical/fixedpoint
    POST /api/matrix/determinant
    POST /api/matrix/inverse
    POST /api/matrix/system
"""

import sys, os, re, json, traceback
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
import sympy as sp
from sympy import latex as sp_latex

# ── Math modules ──────────────────────────────────────────────────────────────
from modules.algebra   import (solve_linear, solve_quadratic,
                                solve_inequality, solve_system)
from modules.analysis  import (compute_limit, compute_derivative,
                                compute_integral, plot_function)
from modules.numerical import (newton_method, bisection_method,
                                fixed_point_iteration)
from modules.matrix    import (compute_determinant, compute_inverse,
                                solve_matrix_system)

app = Flask(__name__)


# =============================================================================
# UTILITIES
# =============================================================================

def preprocess(expr: str) -> str:
    """x^2 → x**2,  2x → 2*x"""
    if not expr:
        return expr
    expr = re.sub(r'\^', '**', expr.strip())
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    return expr


_SP = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}


def to_latex(raw: str) -> str | None:
    """Kiritilgan ifodani LaTeX ga o'tkazadi."""
    if not raw or not raw.strip():
        return None
    proc = preprocess(raw)
    try:
        for sym, cls in [('>=', sp.Ge), ('<=', sp.Le),
                         ('>',  sp.Gt), ('<',  sp.Lt)]:
            if sym in proc:
                l, r = proc.split(sym, 1)
                return sp_latex(cls(sp.sympify(l.strip(), locals=_SP),
                                    sp.sympify(r.strip(), locals=_SP)))
        if '=' in proc:
            l, r = proc.split('=', 1)
            return sp_latex(sp.Eq(sp.sympify(l.strip(), locals=_SP),
                                  sp.sympify(r.strip(), locals=_SP)))
        return sp_latex(sp.sympify(proc, locals=_SP))
    except Exception:
        return None


def to_rational(mat: list) -> list:
    """float list → SymPy Rational list"""
    return [[sp.Rational(v).limit_denominator(1000) for v in row]
            for row in mat]


def serialize_result(result: dict) -> dict:
    """
    SymPy obyektlarini JSON-serializatsiya qilish uchun o'tkazadi.
    answer ni LaTeX ga, steps ni oddiy dict ga.
    """
    if not result:
        return {"error": "Natija bo'sh"}

    out = {
        "problem_type": result.get("problem_type", ""),
        "extra_info":   result.get("extra_info", ""),
        "error":        result.get("error", ""),
        "steps":        [],
        "answer_latex": "",
        "plot":         result.get("plot", ""),
        "table":        result.get("table", []),
    }

    # Steps
    for s in result.get("steps", []):
        step = {
            "step":    s.get("step", 0),
            "title":   s.get("title", ""),
            "content": s.get("content", ""),
            "latex":   s.get("latex", ""),
        }
        out["steps"].append(step)

    # Answer → LaTeX
    ans = result.get("answer")
    try:
        if ans is None:
            out["answer_latex"] = ""
        elif isinstance(ans, (list, tuple)):
            parts = []
            for i, a in enumerate(ans):
                try:
                    parts.append(f"x_{{{i+1}}} = {sp_latex(a)}")
                except Exception:
                    parts.append(str(a))
            out["answer_latex"] = r",\quad ".join(parts)
        elif isinstance(ans, dict):
            parts = [f"{sp_latex(k)} = {sp_latex(v)}"
                     for k, v in ans.items()]
            out["answer_latex"] = r",\quad ".join(parts)
        elif isinstance(ans, sp.Matrix):
            out["answer_latex"] = sp_latex(ans)
        elif isinstance(ans, sp.Basic):
            out["answer_latex"] = sp_latex(ans)
        elif isinstance(ans, (int, float)):
            out["answer_latex"] = f"{ans:.10g}"
        else:
            try:
                out["answer_latex"] = sp_latex(sp.sympify(str(ans)))
            except Exception:
                out["answer_latex"] = str(ans)
    except Exception:
        out["answer_latex"] = str(ans)

    return out


def api_handler(func, *args, **kwargs):
    """Barcha API handlerlar uchun try/except wrapper."""
    try:
        result = func(*args, **kwargs)
        return jsonify(serialize_result(result))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "steps": [],
                        "answer_latex": "", "plot": ""}), 500


# =============================================================================
# MAIN PAGE
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# =============================================================================
# LATEX PREVIEW ENDPOINT
# =============================================================================

@app.route("/api/latex", methods=["POST"])
def latex_preview():
    data = request.get_json(force=True)
    raw  = data.get("expr", "")
    ltx  = to_latex(raw)
    return jsonify({"latex": ltx or ""})


# =============================================================================
# ALGEBRA ROUTES
# =============================================================================

@app.route("/api/algebra/linear", methods=["POST"])
def api_linear():
    d = request.get_json(force=True)
    return api_handler(solve_linear,
                       preprocess(d.get("equation", "")),
                       d.get("variable", "x"))


@app.route("/api/algebra/quadratic", methods=["POST"])
def api_quadratic():
    d = request.get_json(force=True)
    return api_handler(solve_quadratic,
                       preprocess(d.get("equation", "")),
                       d.get("variable", "x"))


@app.route("/api/algebra/inequality", methods=["POST"])
def api_inequality():
    d = request.get_json(force=True)
    return api_handler(solve_inequality,
                       preprocess(d.get("inequality", "")),
                       d.get("variable", "x"))


@app.route("/api/algebra/system", methods=["POST"])
def api_system():
    d = request.get_json(force=True)
    eqs  = [preprocess(e) for e in d.get("equations", [])]
    vars_ = d.get("variables", ["x", "y"])
    return api_handler(solve_system, eqs, vars_)


# =============================================================================
# ANALYSIS ROUTES
# =============================================================================

@app.route("/api/analysis/limit", methods=["POST"])
def api_limit():
    d = request.get_json(force=True)
    return api_handler(compute_limit,
                       preprocess(d.get("function", "")),
                       preprocess(d.get("point", "0")),
                       d.get("direction", "+-"),
                       d.get("variable", "x"))


@app.route("/api/analysis/derivative", methods=["POST"])
def api_derivative():
    d = request.get_json(force=True)
    return api_handler(compute_derivative,
                       preprocess(d.get("function", "")),
                       d.get("variable", "x"),
                       int(d.get("order", 1)))


@app.route("/api/analysis/integral", methods=["POST"])
def api_integral():
    d    = request.get_json(force=True)
    lo   = preprocess(d.get("lower", "")) or None
    hi   = preprocess(d.get("upper", "")) or None
    return api_handler(compute_integral,
                       preprocess(d.get("function", "")),
                       d.get("variable", "x"),
                       lower=lo, upper=hi)


@app.route("/api/analysis/plot", methods=["POST"])
def api_plot():
    d = request.get_json(force=True)
    return api_handler(plot_function,
                       preprocess(d.get("function", "")),
                       float(d.get("xmin", -10)),
                       float(d.get("xmax", 10)),
                       d.get("variable", "x"))


# =============================================================================
# NUMERICAL ROUTES
# =============================================================================

@app.route("/api/numerical/newton", methods=["POST"])
def api_newton():
    d = request.get_json(force=True)
    return api_handler(newton_method,
                       preprocess(d.get("function", "")),
                       float(d.get("x0", 1.0)),
                       float(d.get("tol", 1e-7)),
                       int(d.get("max_iter", 50)))


@app.route("/api/numerical/bisection", methods=["POST"])
def api_bisection():
    d = request.get_json(force=True)
    return api_handler(bisection_method,
                       preprocess(d.get("function", "")),
                       float(d.get("a", 0)),
                       float(d.get("b", 2)),
                       float(d.get("tol", 1e-6)),
                       int(d.get("max_iter", 60)))


@app.route("/api/numerical/fixedpoint", methods=["POST"])
def api_fixedpoint():
    d = request.get_json(force=True)
    return api_handler(fixed_point_iteration,
                       preprocess(d.get("phi", "")),
                       float(d.get("x0", 1.0)),
                       float(d.get("tol", 1e-7)))


# =============================================================================
# MATRIX ROUTES
# =============================================================================

@app.route("/api/matrix/determinant", methods=["POST"])
def api_determinant():
    d   = request.get_json(force=True)
    mat = to_rational(d.get("matrix", [[1,0],[0,1]]))
    return api_handler(compute_determinant, mat)


@app.route("/api/matrix/inverse", methods=["POST"])
def api_inverse():
    d   = request.get_json(force=True)
    mat = to_rational(d.get("matrix", [[1,0],[0,1]]))
    return api_handler(compute_inverse, mat)


@app.route("/api/matrix/system", methods=["POST"])
def api_matrix_system():
    d    = request.get_json(force=True)
    A    = to_rational(d.get("A", [[1,0],[0,1]]))
    b    = to_rational(d.get("b", [[0],[0]]))
    return api_handler(solve_matrix_system, A, b)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Production: Render/Railway use PORT env variable
    # Local:       default port 5000
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print(f"\n  MathSolverPy starting on port {port}  (debug={debug})\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
