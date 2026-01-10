
import deepgram
import inspect

def explore(name, obj, depth=0):
    if depth > 2: return
    try:
        print(f"{'  '*depth}Module: {name}")
        for attr in dir(obj):
            if attr.startswith("_"): continue
            val = getattr(obj, attr)
            if "Options" in attr or "Events" in attr or "Microphone" in attr:
                print(f"{'  '*(depth+1)}Found: {attr}")
            if inspect.ismodule(val) and "deepgram" in val.__name__:
                 explore(attr, val, depth+1)
    except Exception as e:
        print(f"Error exploring {name}: {e}")

explore("deepgram", deepgram)
