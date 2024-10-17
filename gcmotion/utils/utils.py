import sys
import importlib


def reload():
    for mod in [mod for name, mod in sys.modules.items() if "gcmotion" in name]:
        importlib.reload(mod)
