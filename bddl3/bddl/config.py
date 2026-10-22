"""BDDL library configuration: file paths and constants.

This module centralizes the filesystem layout for BDDL activity definitions
and domain files, and provides helpers to resolve them by name.

Package data (activity definitions, generated data) should be accessed through
the :class:`importlib.resources.abc.Traversable` objects defined here
(:data:`PACKAGE_ROOT`, :data:`ACTIVITY_CONFIGS_DIR`, :data:`GENERATED_DATA_DIR`)
and the ``get_*_resource`` helpers, using only ``Traversable`` methods
(``/``, ``joinpath``, ``open``, ``read_text``, ``read_bytes``, ``iterdir``,
``is_dir``, ``is_file``, ``name``).  This keeps BDDL working when it is
imported from a zip archive (e.g. a zipapp or other zipimport-based bundle),
where the package directory is not a real filesystem directory and APIs like
``open()``, ``os.listdir()`` or ``glob.glob()`` cannot be used.
"""

import importlib.resources
import os

# Root of the installed ``bddl`` package as a Traversable.  This is a
# ``pathlib.Path`` for regular installs and a ``zipfile.Path`` when BDDL is
# imported from a zip archive.
PACKAGE_ROOT = importlib.resources.files("bddl")

# Traversable directory containing activity problem files and domain files.
ACTIVITY_CONFIGS_DIR = PACKAGE_ROOT / "activity_definitions"

# Traversable directory containing generated data (taxonomy, transition maps,
# knowledge base source data, etc.).
GENERATED_DATA_DIR = PACKAGE_ROOT / "generated_data"

# Absolute path to the directory containing activity problem files and domain
# files.  Each activity has a subdirectory with one or more ``problemN.bddl``
# files, and domain files live directly inside this directory.
#
# NOTE: This is a plain filesystem path and is NOT usable when BDDL is imported
# from a zip archive.  Prefer :data:`ACTIVITY_CONFIGS_DIR` and the
# ``get_*_resource`` helpers for reading package data.
ACTIVITY_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "activity_definitions")

# BDDL requirement strings understood by the parser (subset of PDDL).
SUPPORTED_BDDL_REQUIREMENTS = [":strips", ":negative-preconditions", ":typing", ":adl"]

# Human-readable names for predicates that are awkward when used directly
# (e.g. "ontop" -> "on top of").
READABLE_PREDICATE_NAMES = {"ontop": "on top of", "nextto": "next to"}


def get_definition_resource(behavior_activity, instance, domain=False):
    """Return a Traversable for a BDDL problem or domain file.

    Unlike :func:`get_definition_filename`, the returned object can be read
    (via ``.read_text()``, ``.read_bytes()`` or ``.open()``) even when BDDL is
    imported from a zip archive.

    Args:
        behavior_activity: Activity name (ignored when *domain* is True).
        instance: Integer definition index (e.g. ``0``).
        domain: If True, return the legacy domain file
            (``domain_behavior-100.bddl``) rather than a problem file.

    Returns:
        importlib.resources.abc.Traversable: The requested ``.bddl`` file.
    """
    if domain:
        return ACTIVITY_CONFIGS_DIR / "domain_behavior-100.bddl"
    else:
        return ACTIVITY_CONFIGS_DIR / behavior_activity / f"problem{instance}.bddl"


def get_domain_resource(domain_name):
    """Return a Traversable for a BDDL domain file by name.

    Unlike :func:`get_domain_filename`, the returned object can be read even
    when BDDL is imported from a zip archive.

    Args:
        domain_name: Short name of the domain (e.g. ``"behavior-1k"``).

    Returns:
        importlib.resources.abc.Traversable: The domain ``.bddl`` file.
    """
    return ACTIVITY_CONFIGS_DIR / f"domain_{domain_name}.bddl"


def get_definition_filename(behavior_activity, instance, domain=False):
    """Return the filesystem path to a BDDL problem or domain file.

    NOTE: The returned path cannot be opened directly when BDDL is imported
    from a zip archive; prefer :func:`get_definition_resource` for reading.

    Args:
        behavior_activity: Activity name (ignored when *domain* is True).
        instance: Integer definition index (e.g. ``0``).
        domain: If True, return the path to the legacy domain file
            (``domain_behavior-100.bddl``) rather than a problem file.

    Returns:
        str: Absolute path to the requested ``.bddl`` file.
    """
    if domain:
        return os.path.join(ACTIVITY_CONFIGS_PATH, "domain_behavior-100.bddl")
    else:
        return os.path.join(
            ACTIVITY_CONFIGS_PATH, behavior_activity, f"problem{instance}.bddl"
        )


def get_domain_filename(domain_name):
    """Return the filesystem path to a BDDL domain file by name.

    NOTE: The returned path cannot be opened directly when BDDL is imported
    from a zip archive; prefer :func:`get_domain_resource` for reading.

    The domain file is expected at
    ``<ACTIVITY_CONFIGS_PATH>/domain_<domain_name>.bddl``.

    Args:
        domain_name: Short name of the domain (e.g. ``"behavior-1k"``).

    Returns:
        str: Absolute path to the domain ``.bddl`` file.
    """
    return os.path.join(ACTIVITY_CONFIGS_PATH, f"domain_{domain_name}.bddl")
