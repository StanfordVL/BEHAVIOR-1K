"""Lightweight CLI helpers for Newton-native examples."""

import random


def choose_from_options(options, name, random_selection=False):
    """Choose an option without importing the legacy OmniGibson UI stack."""
    options = list(options)
    if not options:
        raise ValueError(f"No options available for {name}.")
    if random_selection:
        return random.choice(options)

    for idx, option in enumerate(options):
        print(f"{idx}: {option}")
    try:
        selection = input(f"Choose {name}: ")
    except EOFError:
        try:
            with open("/dev/tty") as tty:
                print(f"Choose {name}: ", end="", flush=True)
                selection = tty.readline()
        except OSError as exc:
            raise RuntimeError(
                "Interactive selection needs a terminal. If using conda run, pass --no-capture-output."
            ) from exc
    selection = selection.strip() or "0"
    try:
        return options[int(selection)]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid {name} selection: {selection!r}") from exc
