# MathSolverPy

Pythonda algebra va analiz masalalarini yechish uchun akademik dasturiy ta'minot.
Magistrlik dissertatsiyasi loyihasi — Flask + SymPy + MathJax 3.

## Lokal ishga tushirish

    pip install -r requirements.txt
    python server.py
    # http://localhost:5000

## Texnologiyalar

Python 3.11, Flask 3.1, SymPy, NumPy, SciPy, Matplotlib, MathJax 3

## Deploy

    gunicorn server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
