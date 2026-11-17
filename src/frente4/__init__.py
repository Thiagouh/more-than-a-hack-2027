"""Frente 4: baseline reproduzido e máquina de estados de posse (H/A/D/X por frame).

Entrada: tracking ATD (frente 1), Smart Tagging (frente 3) e, opcionalmente, eventos de
toque/controle (frente 5). Saída: CSV ``frame,label`` com 199.319 linhas, gerado só por código.

    python -m src.frente4 run --out outputs/submissions/frente4.csv
    python -m src.frente4 baseline --out outputs/submissions/baselines
    python -m src.frente4 from-csv evaluation/results/baseline_raw.csv --out outputs/submissions/dev.csv

Nenhum módulo lê o gabarito. A janela pública serve só para conferir a integração.
"""
from .fsm import possession_labels, Params

__all__ = ["possession_labels", "Params"]
