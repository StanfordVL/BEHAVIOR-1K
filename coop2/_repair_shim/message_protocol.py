"""Shared COOP2 repair-message protocol constants.

Copied verbatim from ma_crafter: this file is only string constants and carries
no repair logic, so there is nothing to stub. The message *type* still has to
match upstream's, because the plan wrapper filters the broker's message log on
it -- a divergent constant would silently stop filtering rather than error.
"""

COOP2_REPAIR_MESSAGE_TYPE = "coop2_repair_request"
COOP2_REPAIR_CONTENT_TYPE = "coop2_pre_execution_repair"
COOP2_REPAIR_SENDER_ID = "coop2_repair"
MESSAGE_TYPE_METADATA_KEY = "message_type"


def coop2_repair_metadata() -> dict:
    """Return metadata for a COOP2 repair request message."""
    return {MESSAGE_TYPE_METADATA_KEY: COOP2_REPAIR_MESSAGE_TYPE}
