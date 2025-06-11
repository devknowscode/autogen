"""
This module provides utility functions for the autogen-custom package.
It includes functions for patching modules at runtime.
"""

from .patch import patch_module

__all__ = ["patch_module"]
