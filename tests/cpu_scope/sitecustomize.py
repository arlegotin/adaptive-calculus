"""Explicit pytest CPU scope for fresh interpreter replay; no scientific proof."""
import os
if os.environ.get('RALUP_CPU_TESTS') == '1':
    import functools
    import sys
    from importlib.machinery import PathFinder

    class CPUScopeLoader:
        def __init__(self, loader):
            self.loader = loader
        def __getattr__(self, name):
            return getattr(self.loader, name)
        def create_module(self, spec):
            return self.loader.create_module(spec)
        def exec_module(self, module):
            self.loader.exec_module(module)
            module.is_available = functools.lru_cache(maxsize=None)(lambda: False)

    class CPUScopeFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'torch.backends.mps':
                spec = PathFinder.find_spec(fullname, path)
                spec.loader = CPUScopeLoader(spec.loader)
                return spec

    sys.meta_path.insert(0, CPUScopeFinder())
