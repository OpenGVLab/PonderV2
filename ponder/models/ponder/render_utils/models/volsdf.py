from functools import partial

from ..builder import RENDERERS
from .base_surface_model import SurfaceModel


@RENDERERS.register_module()
class VolSDFModel(SurfaceModel):
    def __init__(self, field, collider, sampler, loss, **kwargs):
        super().__init__(field=field, collider=collider, sampler=sampler, loss=loss)

    def sample_and_forward_field(self, ray_bundle, volume_feature):
        sampler_out_dict = self.sampler(
            ray_bundle,
            density_fn=self.field.laplace_density,
            sdf_fn=partial(self.field.get_sdf, volume_feature=volume_feature),
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, volume_feature)
        weights, _ = ray_samples.get_weights_and_transmittance(field_outputs["density"])

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),  # (num_rays, num_smaples+num_importance, 3)
            **sampler_out_dict,
        }
        return samples_and_field_outputs
