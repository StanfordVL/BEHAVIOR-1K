from bddl.knowledge_base import SynsetState


def tocolor_filter(state):
    """Convert a SynsetState to a Bootstrap color class."""
    color_map = {
        SynsetState.MATCHED: "success",
        SynsetState.PLANNED: "warning",
        SynsetState.UNMATCHED: "danger",
        SynsetState.ILLEGAL: "secondary",
        SynsetState.NONE: "light",
    }
    return color_map.get(state, "light")


# Standalone functions for static site generator
def status_color(state):
    """Convert a SynsetState to a Bootstrap color class (for static generator)."""
    color_map = {
        SynsetState.MATCHED: "success",
        SynsetState.PLANNED: "warning",
        SynsetState.UNMATCHED: "danger",
        SynsetState.ILLEGAL: "secondary",
        SynsetState.NONE: "light",
    }
    return color_map.get(state, "light")


def status_color_transition_rule(state):
    """Convert a TransitionRule state to a Bootstrap color class (for static generator)."""
    # For now, using the same mapping as status_color
    # This can be customized if TransitionRule states differ
    return status_color(state)
