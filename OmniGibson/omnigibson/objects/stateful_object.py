# StatefulObject has been merged into BaseObject.
# This shim exists only for backwards compatibility with any external code that imports StatefulObject.
from omnigibson.objects.object_base import BaseObject as StatefulObject

__all__ = ["StatefulObject"]
