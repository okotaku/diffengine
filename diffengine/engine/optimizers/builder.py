from diffengine.registry import OPTIMIZERS

try:
    import apex
except ImportError:
    apex = None

def register_apex_optimizers() -> list:
    """Register transformer optimizers."""
    apex_optimizers = []
    if apex is not None:
        from apex.optimizers import FusedAdam, FusedSGD
        OPTIMIZERS.register_module(name="FusedAdam")(FusedAdam)
        apex_optimizers.append("FusedAdam")
        OPTIMIZERS.register_module(name="FusedSGD")(FusedSGD)
        apex_optimizers.append("FusedSGD")
    return apex_optimizers


APEX_OPTIMIZERS = register_apex_optimizers()
