import importlib
import inspect


def patch_module(module_path: str, func_name: str, new_func):
    """
    Replace module_path.func_name with new_func at runtime.

    Args:
        module_path: e.g. 'autogen.some_module'
        func_name:   name of the function to replace in that module
        new_func:    a callable with a compatible signature
    """
    # 1. Import (or reload) the target module
    module = importlib.import_module(module_path)

    # 2. Verify the module actually has that attribute
    if not hasattr(module, func_name):
        raise AttributeError(f"{module_path} has no attribute {func_name!r}")

    # 3. Optionally check that new_func is callable
    if not callable(new_func):
        raise TypeError(f"new_func must be callable, got {type(new_func)}")

    # 4. (Optional) Verify signature compatibility
    orig = getattr(module, func_name)
    if inspect.isfunction(orig):
        sig_orig = inspect.signature(orig)
        sig_new  = inspect.signature(new_func)
        # Simple check: same number of parameters
        if len(sig_new.parameters) != len(sig_orig.parameters):
            raise ValueError(
                f"Signature mismatch: {func_name}{sig_new} vs original {sig_orig}"
            )

    # 5. Monkey-patch
    setattr(module, func_name, new_func)
