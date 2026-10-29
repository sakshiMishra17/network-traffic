"""A tiny fake of the `streamlit` module. It records every call so we can run
app/dashboard.py top-to-bottom without Streamlit installed (smoke test of OUR logic).
It cannot check that real Streamlit accepts our arguments."""
import types


class StopScript(Exception):
    pass


class _Ctx:
    """Works as a context manager and swallows any method call (metric, markdown, ...)."""
    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        def f(*args, **kwargs):
            self._log.append(name)
        return f


def make_fake(scenario, log):
    """scenario keys: source, model (or None), uploaded (file-like or None), checkbox"""
    st = types.ModuleType("streamlit")
    st.log = log

    def rec(name, ret=None):
        def f(*a, **k):
            log.append(name if name != "error" else ("error", str(a[0]) if a else ""))
            return ret
        return f

    for n in ["set_page_config", "title", "caption", "header", "subheader", "markdown",
              "info", "warning", "error", "metric", "pyplot", "dataframe", "image",
              "download_button", "write", "text"]:
        setattr(st, n, rec(n))

    def cache_resource(*a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return lambda fn: fn
    st.cache_resource = cache_resource

    def stop():
        log.append("stop")
        raise StopScript()
    st.stop = stop
    st.spinner = lambda *a, **k: _Ctx(log)
    st.columns = lambda n, *a, **k: [_Ctx(log) for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda names: [_Ctx(log) for _ in names]
    st.checkbox = lambda *a, **k: scenario.get("checkbox", False)

    class Sidebar(_Ctx):
        def selectbox(self, label, options, format_func=None, **k):
            options = list(options)
            if format_func:
                for o in options:
                    format_func(o)          # make sure the formatter itself works
            m = scenario.get("model")
            return m if m in options else options[0]

        def radio(self, label, options, **k):
            return scenario.get("source", options[0])

        def file_uploader(self, *a, **k):
            return scenario.get("uploaded")

    st.sidebar = Sidebar(log)
    return st
