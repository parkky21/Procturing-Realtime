import sys
import os

# Mock LiveKit imports as they might not be installed in this env
# If they are, great. If not, we mock them to verify the *rest* of the imports.
try:
    import livekit
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["livekit"] = MagicMock()
    sys.modules["livekit.rtc"] = MagicMock()
    print("Mocked LiveKit modules.")

try:
    from app.services.procturing import Procturing
    print("Successfully imported Procturing class.")
except Exception as e:
    print(f"Import Error: {e}")
    import traceback
    traceback.print_exc()
