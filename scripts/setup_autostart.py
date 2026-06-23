"""
One-time setup: register JUTSU to auto-start at Windows login.

  Register:  python scripts/setup_autostart.py
  Remove:    python scripts/setup_autostart.py --remove
"""
import os
import sys
import winreg

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHONW = os.path.join(ROOT, '.venv', 'Scripts', 'pythonw.exe')
MAIN    = os.path.join(ROOT, 'watcher.py')
REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def register():
    if not os.path.exists(PYTHONW):
        print(f"ERROR: {PYTHONW} not found — set up the venv first.")
        sys.exit(1)

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "JUTSU", 0, winreg.REG_SZ, f'"{PYTHONW}" "{MAIN}"')
        print(f"Registered JUTSU: {PYTHONW}")

    print("\nJUTSU will auto-start silently on next login.")
    print("Requires UnityCapture to be installed (run Install.bat as Admin, then reboot).")
    print("Run with --remove to undo.")


def remove():
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_KEY, 0, winreg.KEY_SET_VALUE) as k:
        try:
            winreg.DeleteValue(k, "JUTSU")
            print("Removed JUTSU.")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    if "--remove" in sys.argv:
        remove()
    else:
        register()
