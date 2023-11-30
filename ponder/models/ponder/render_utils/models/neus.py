from functools import partial

from ..builder import RENDERERS
from .base_surface_model import SurfaceModel


@RENDERERS.register_module()
class NeuSModel(SurfaceModel):
    def __init__(self, field, collider, sampler, loss, **kwargs):
        super().__init__(field=field, collider=collider, sampler=sampler, loss=loss)
        self.anneal_end = 50000

    def get_training_callbacks(self):
        raise NotImplementedError

    def sample_and_forward_field(self, ray_bundle, volume_feature):
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(self.field.get_sdf, volume_feature=volume_feature),
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, volume_feature, return_alphas=True)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs["alphas"]
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,  # (num_rays, num_smaples+num_importance, 1)
            "sampled_points": ray_samples.frustums.get_start_positions(),  # (num_rays, num_smaples+num_importance, 3)
            **sampler_out_dict,
        }

        return samples_and_field_outputs
