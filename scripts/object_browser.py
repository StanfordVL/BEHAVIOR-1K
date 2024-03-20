import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
import numpy as np

from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)

from b1k_pipeline.usd_conversion.import_metadata import import_obj_metadata

dataset_root = "/scr/OmniGibson/omnigibson/data/og_dataset"

all_categories = get_all_object_categories()
for cat in all_categories:
    print(f"Now processing category {cat}")
    models = get_all_object_category_models(cat)
    # randomly pick a model
    model = np.random.choice(models)
    print(f"Randomly picked model {model}")
    # import metadata
    import_obj_metadata(cat, model, dataset_root)
