from monai.transforms import MapTransform, LoadImaged
from monai.data import MetaTensor
import numpy as np
import os


class MergeTotalSegLabelsd(MapTransform):
    def __init__(self, keys, label_names):
        super().__init__(keys)
        self.label_names = label_names
        self.loader = LoadImaged(keys=["temp_key"], image_only=False)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # 1. Get the first available path to use as a template
            first_organ_path = next(path for path in d[key].values() if path is not None)

            # 2. Load the reference image
            ref_dict = self.loader({"temp_key": first_organ_path})
            ref_data = ref_dict["temp_key"]

            # 3. Create the zero-filled array with channel dimension
            # Ensure we have shape (1, H, W, D) for labels
            if len(ref_data.shape) == 3:  # If no channel dim
                combined_label = np.zeros((1,) + ref_data.shape, dtype=np.float32)
                ref_shape_with_channel = (1,) + ref_data.shape
            else:  # If already has channel dim
                combined_label = np.zeros(ref_data.shape, dtype=np.float32)
                ref_shape_with_channel = ref_data.shape

            for idx, organ in enumerate(self.label_names, start=1):
                path = d[key].get(organ)
                if path and os.path.exists(path):
                    organ_dict = self.loader({"temp_key": path})
                    organ_data = organ_dict["temp_key"]

                    # Handle organ data shape
                    if len(organ_data.shape) == 3:  # No channel dim
                        organ_data_2d = organ_data
                    else:  # Has channel dim
                        organ_data_2d = organ_data[0] if organ_data.shape[0] == 1 else organ_data

                    # Apply the label
                    if len(combined_label.shape) == 4:  # Has channel dim
                        combined_label[0][organ_data_2d > 0.5] = idx
                    else:  # No channel dim (shouldn't happen)
                        combined_label[organ_data_2d > 0.5] = idx

            # 4. Wrap it back into a MetaTensor
            if isinstance(ref_data, MetaTensor):
                # Ensure we have the affine from the original data
                meta_dict = {}
                if hasattr(ref_data, 'meta'):
                    meta_dict = ref_data.meta.copy()

                # Create MetaTensor with proper metadata
                d[key] = MetaTensor(
                    combined_label,
                    affine=ref_data.affine,
                    meta=meta_dict
                )
            else:
                d[key] = combined_label

        return d