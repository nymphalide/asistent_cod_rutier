import sys
import asyncio

def setup_windows_asyncio():
    """Fixes Event Loop policy on Windows/WSL2"""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())