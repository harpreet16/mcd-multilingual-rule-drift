"""Figures and LaTeX tables made from the results.

Input:  results.json, plus grades.jsonl for the rule grid (labels come from metrics.label_rule)
Output: image files and .tex tables in a folder

One figure is a grid of rule against turn: green for PASS, red for FAIL, with the rule's label beside each row.
Start from the earlier figures (retention, trajectory, first violation turn, leakage). One command makes everything.
"""


def make_figures(results_path, output_folder):
    """Write every figure and table into the output folder."""
    raise NotImplementedError("Not built yet")
