from flask import Blueprint
from bddl.knowledge_base import SynsetState

bp = Blueprint('filters', __name__)

@bp.app_template_filter('tocolor')
def tocolor_filter(state):
    """Convert a SynsetState to a Bootstrap color class."""
    color_map = {
        SynsetState.MATCHED: "success",
        SynsetState.PLANNED: "warning", 
        SynsetState.UNMATCHED: "danger",
        SynsetState.ILLEGAL: "secondary",
        SynsetState.NONE: "light"
    }
    return color_map.get(state, "light")
