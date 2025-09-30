def test_import():
    # Import-only smoke check (mirrors your README's "Import-only smoke test")
    # https://github.com/eniktab/GASAL2-Py/blob/main/README.md
    import gasal2
    assert hasattr(gasal2, "GasalAligner")
