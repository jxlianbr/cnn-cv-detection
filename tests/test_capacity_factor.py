"""The area-to-capacity factor lives in one place (DEFAULT_KW_PER_M2) and every
script takes it from there. A stale hard-coded 0.15 kW/m^2 once left two logs
binned with the wrong factor; these tests keep that from happening again."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.failure_cases import (DEFAULT_KW_PER_M2, capacity_bin_masks,
                                          evaluate_saved_predictions)

ROOT = Path(__file__).resolve().parents[1]
CAPACITY_SCRIPTS = ["run_failure_case_evaluation.py", "run_size_detectability.py",
                    "make_figures.py", "make_annotation_sample.py",
                    "run_revision_checks.py"]
CAPACITY_FUNCTIONS = {"capacity_bin_masks": 2, "estimate_capacity_kw": 1,
                      "flag_sub_50kw": 1, "build_subset_masks": 1,
                      "stratified_annotation_sample": 3}  # positional index of kw_per_m2


def _sources():
    files = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
    for path in files:
        try:
            yield path, ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a file mid-edit is someone else's problem
            continue


def _is_number(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _hard_coded_factors(tree):
    """(line, description) for every numeric literal passed or defaulted as a
    capacity factor."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "kw_per_m2" and _is_number(kw.value):
                    yield node.lineno, f"kw_per_m2={kw.value.value}"
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name in CAPACITY_FUNCTIONS:
                i = CAPACITY_FUNCTIONS[name]
                if len(node.args) > i and _is_number(node.args[i]):
                    yield node.lineno, f"{name}(..., {node.args[i].value})"
            is_kw_flag = (getattr(node.func, "attr", None) == "add_argument"
                          and any(isinstance(a, ast.Constant) and a.value == "--kw-per-m2"
                                  for a in node.args))
            if is_kw_flag:
                default = next((k.value for k in node.keywords if k.arg == "default"), None)
                if not (isinstance(default, ast.Name) and default.id == "DEFAULT_KW_PER_M2"):
                    yield node.lineno, "--kw-per-m2 default is not DEFAULT_KW_PER_M2"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + node.args.defaults + node.args.kw_defaults)
            for arg, default in zip(args, defaults):
                if arg.arg == "kw_per_m2" and default is not None and _is_number(default):
                    yield node.lineno, f"def {node.name}(kw_per_m2={default.value})"


def test_default_factor_is_the_label_sources_conversion():
    # 0.86 MW/ha (Shimada & Takeuchi 2025): 50 kW is ~580 m^2
    assert DEFAULT_KW_PER_M2 == pytest.approx(0.086)
    assert 50 / DEFAULT_KW_PER_M2 == pytest.approx(581.4, abs=0.1)


def test_no_script_or_module_hard_codes_a_capacity_factor():
    offenders = [f"{path.relative_to(ROOT)}:{line}: {what}"
                 for path, tree in _sources() for line, what in _hard_coded_factors(tree)]
    assert not offenders, "use DEFAULT_KW_PER_M2 instead:\n" + "\n".join(offenders)


@pytest.mark.parametrize("script", CAPACITY_SCRIPTS)
def test_capacity_scripts_use_the_shared_constant(script):
    path = ROOT / "scripts" / script
    if not path.exists():
        pytest.skip(f"{script} not present")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "src.evaluation.failure_cases" for alias in node.names}
    assert "DEFAULT_KW_PER_M2" in imported


def test_the_detector_flags_a_hard_coded_factor():
    tree = ast.parse("capacity_bin_masks(df, edges, 0.15)\n"
                     "estimate_capacity_kw(a, kw_per_m2=0.15)\n"
                     "parser.add_argument('--kw-per-m2', type=float, default=0.15)\n"
                     "def f(x, kw_per_m2=0.15): pass\n"
                     "capacity_bin_masks(df, edges, DEFAULT_KW_PER_M2)\n")
    assert sorted(line for line, _ in _hard_coded_factors(tree)) == [1, 2, 3, 4]


def _predictions():
    # areas in m^2 -> kW at 0.086: 500 -> 43, 2000 -> 172, 20000 -> 1720
    return pd.DataFrame({
        "point_id": [1, 2, 3, 4],
        "label": ["pv", "pv", "pv", "non_pv"],
        "polygon_area_m2": [500.0, 2000.0, 20000.0, np.nan],
        "xgboost_baseline_proba": [0.9, 0.2, 0.8, 0.1],
        "cnn_proba_seed1": [0.1, 0.9, 0.9, 0.6],
        "cnn_proba_seed0": [0.6, 0.6, 0.9, 0.2],
    })


def test_evaluate_saved_predictions_reports_every_run_in_seed_order():
    pred = _predictions()
    masks = {"all": np.ones(len(pred), dtype=bool)}
    report = evaluate_saved_predictions(pred, masks)
    assert report[["model", "repeat"]].values.tolist() == [
        ["xgboost_baseline", 0], ["cnn", 0], ["cnn", 1]]
    assert report["n"].tolist() == [4, 4, 4]
    assert report["recall"].tolist() == pytest.approx([2 / 3, 1.0, 2 / 3])


def test_evaluate_saved_predictions_bins_with_the_default_factor():
    pred = _predictions()
    masks = capacity_bin_masks(pred, [0, 50, 500, np.inf])
    report = evaluate_saved_predictions(pred, masks)
    seed0 = report[(report["model"] == "cnn") & (report["repeat"] == 0)]
    # unsized non-PV point belongs to no bin; 500 m^2 is under 50 kW at 0.086
    assert seed0.set_index("subset")["n"].to_dict() == {"0-50kW": 1, "50-500kW": 1,
                                                        ">500kW": 1}
