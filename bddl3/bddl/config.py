"""BDDL library configuration: file paths and constants.

This module centralizes the filesystem layout for BDDL activity definitions
and domain files, and provides helpers to resolve them by name.
"""

import os

# Absolute path to the directory containing activity problem files and domain
# files.  Each activity has a subdirectory with one or more ``problemN.bddl``
# files, and domain files live directly inside this directory.
ACTIVITY_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "activity_definitions")

# BDDL requirement strings understood by the parser (subset of PDDL).
SUPPORTED_BDDL_REQUIREMENTS = [":strips", ":negative-preconditions", ":typing", ":adl"]

# Human-readable names for predicates that are awkward when used directly
# (e.g. "ontop" -> "on top of").
READABLE_PREDICATE_NAMES = {"ontop": "on top of", "nextto": "next to"}


def get_definition_filename(behavior_activity, instance, domain=False):
    """Return the filesystem path to a BDDL problem or domain file.

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

    The domain file is expected at
    ``<ACTIVITY_CONFIGS_PATH>/domain_<domain_name>.bddl``.

    Args:
        domain_name: Short name of the domain (e.g. ``"behavior-1k"``).

    Returns:
        str: Absolute path to the domain ``.bddl`` file.
    """
    return os.path.join(ACTIVITY_CONFIGS_PATH, f"domain_{domain_name}.bddl")
