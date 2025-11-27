"""
Markdown report generation utilities.

This package contains helpers to summarize experiment results and
generated figures into human-readable Markdown reports under the
top-level ``reports/`` directory.
"""

from .markdown import generate_study_reports, generate_summary_report

__all__ = ["generate_summary_report", "generate_study_reports"]
