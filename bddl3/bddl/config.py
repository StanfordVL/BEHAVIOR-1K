import os

# PATHS
ACTIVITY_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "activity_definitions")

# BDDL
SUPPORTED_BDDL_REQUIREMENTS = [":strips", ":negative-preconditions", ":typing", ":adl"]

READABLE_PREDICATE_NAMES = {"ontop": "on top of", "nextto": "next to"}


def get_definition_filename(behavior_activity, instance, domain=False):
    if domain:
        return os.path.join(ACTIVITY_CONFIGS_PATH, "domain_igibson.bddl")
    else:
        return os.path.join(
            ACTIVITY_CONFIGS_PATH, behavior_activity, f"problem{instance}.bddl"
        )


def get_domain_filename(domain_name):
    return os.path.join(ACTIVITY_CONFIGS_PATH, f"domain_{domain_name}.bddl")
