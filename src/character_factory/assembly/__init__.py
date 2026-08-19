"""Assembly: the deterministic half of the system.

Rig evaluation, rest-pose authoring, UV compositing (arriving with the
texture modules), and the skinned glTF exporter. CPU-capable by design —
this package is what runs on machines that cannot generate.

Needs torch (for the rig's TorchScript forward) and numpy; both are base
dependencies precisely so the assembly-only install story holds
(ARCHITECTURE.md §6.2).
"""

from character_factory.assembly.export import SCALE, ExportResult, export_character_glb
from character_factory.assembly.rig import RigDefinition, load_rig
from character_factory.assembly.validate import validate_glb

__all__ = [
    "ExportResult",
    "RigDefinition",
    "SCALE",
    "export_character_glb",
    "load_rig",
    "validate_glb",
]
