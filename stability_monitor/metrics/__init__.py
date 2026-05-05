"""Per-metric implementations for the stability monitor.

Each module exposes a NumPy ``batch`` function and a ``StreamingX`` class
that produce numerically identical outputs on the same input (modulo causal
lag at stencil boundaries). See ``stability_monitoring_methods.tex``.
"""
