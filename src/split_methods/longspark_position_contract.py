"""Confirm the enabled static position adapter before graph capture."""

FLAGS = {
    "fixed_ntk": "longspark_fixed_ntk_v1",
}


def configured_modes(config):
    if config is None:
        return set()
    enabled = set()
    for mode, flag in FLAGS.items():
        value = getattr(config, flag, False)
        if type(value) is not bool:
            raise RuntimeError(f"Position adapter flag {flag} must be boolean")
        if value:
            enabled.add(mode)
    return enabled


def validate_mode(config, expected):
    if configured_modes(config) != {expected}:
        raise RuntimeError(f"Expected exactly the {expected} position adapter")
