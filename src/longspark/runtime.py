"""Spawn-safe installation of IPC and 14B GQA adapters."""
import os

def install():
    from .registered_transport import install as install_ipc
    install_ipc('split_methods')
    if os.environ.get('LONGSPARK_GQA5') == '1':
        from .gqa14_compat import install as install_gqa
        install_gqa()

def target_process(*args):
    install()
    from split_methods.target import _target_process
    return _target_process(*args)
