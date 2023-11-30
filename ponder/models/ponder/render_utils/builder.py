from ponder.utils.registry import Registry

RENDERERS = Registry("renderers")
FIELDS = Registry("fields")
COLLIDERS = Registry("colliders")
SAMPLERS = Registry("samplers")


def build_renderer(cfg, **kwargs):
    """Build renderers."""
    return RENDERERS.build(cfg, default_args=kwargs)


def build_field(cfg, **kwargs):
    """Build fields."""
    return FIELDS.build(cfg, default_args=kwargs)


def build_collider(cfg, **kwargs):
    """Build colliders."""
    return COLLIDERS.build(cfg, default_args=kwargs)


def build_sampler(cfg, **kwargs):
    """Build samplers."""
    return SAMPLERS.build(cfg, default_args=kwargs)
