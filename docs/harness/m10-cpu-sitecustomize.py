"""Explicit CPU-regression-only scope, including fresh interpreter replays."""
import os
import functools
if os.environ.get('M10_CPU_REGRESSION_SCOPE')=='1':
    import sys
    from importlib.machinery import PathFinder
    class _CPUScopeLoader:
        def __init__(self,loader):self.loader=loader
        def __getattr__(self,name):return getattr(self.loader,name)
        def create_module(self,spec):return self.loader.create_module(spec)
        def exec_module(self,module):
            self.loader.exec_module(module)
            module.is_available=functools.lru_cache(maxsize=None)(lambda:False)
    class _CPUScopeFinder:
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='torch.backends.mps':
                spec=PathFinder.find_spec(fullname,path)
                spec.loader=_CPUScopeLoader(spec.loader)
                return spec
    # Lazy override avoids importing Torch into unrelated supervised utilities.
    sys.meta_path.insert(0,_CPUScopeFinder())
