"""Import official mjlab without loading other users' task entry points.

Only this process's discovery call is filtered. No installed package is changed.
"""
import importlib
from importlib import metadata

def import_mjlab():
    original=metadata.entry_points
    def filtered(*args,**kwargs):
        entries=original(*args,**kwargs)
        return type(entries)(ep for ep in entries if ep.group!='mjlab.tasks')
    metadata.entry_points=filtered
    try:
        return importlib.import_module('mjlab')
    finally:
        metadata.entry_points=original
