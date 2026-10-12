"""Focused regression for named child and administration allocation bounds."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for child,admin,child_ok,admin_ok in [(32,16,True,True),(33,15,False,True),(31,17,True,False)]:
    got=m.allocation_bounds(child,admin,32,16)
    assert got['child_component_bound_pass'] is child_ok,got
    assert got['administration_bound_pass'] is admin_ok,got
    assert got['all_declared_allocation_bounds_pass'] is (child_ok and admin_ok),got
print('PASS: exact boundary; child overrun despite aggregate fit; administration overrun despite aggregate fit')
for resource_ok,floor_ok,exception,expected_exit in [(True,False,True,0),(True,False,False,3),(False,False,True,3),(True,True,False,0)]:
    got=m.route_disposition(resource_ok,floor_ok,exception,0)
    assert got['bounds_pass'] is (resource_ok and floor_ok),got
    assert got['capture_exit_code']==expected_exit,got
    assert got['data_floor_pass'] is floor_ok,got
print('PASS: failed Data floor stays false under scoped zero-write route; unchanged bound failures still stop')
