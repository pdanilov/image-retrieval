"""Named configurations, one per row the README quotes.

The interesting runs carry up to ten flags, and the fine-tuned ones carry a combination
that only works one way -- `--backbone resnet101 --weights sfm120k --p learned --scales
1.0 0.7071067811865476 0.5 --whiten --whiten-source learned` is a lot of surface for a
command anyone is expected to retype. A preset is that command under one name.

They are Python objects rather than YAML for one reason: every config here validates
itself on construction, and several combinations are rejected outright. A YAML preset
would defer that to load time at best and silently mistype a key at worst, which is the
same class of error the validation was added to prevent. As instances, a broken preset
fails when the module is imported -- which the test suite does.

Presets stay overridable: `cbir evaluate gem-ft-r101 --scales 1.0` runs the preset at a
single scale, re-validating as it goes. What they are *not* is a second source of truth
about results -- `results/runs.jsonl` remains that, and `tests/configs/test_presets.py`
holds each preset to a row already recorded there.
"""

from __future__ import annotations

from cbir.configs.classic import BoWConfig, FisherConfig, VLADConfig
from cbir.configs.cnn import NeuralCodesConfig, PooledConfig, RMACConfig
from cbir.configs.run import DescriptorConfig
from cbir.descriptors.cnn.pooling import MULTI_SCALE

PRESETS: dict[str, DescriptorConfig] = {
    # --- classic tier: the best k for each aggregator ---------------------------------
    "bow-best": BoWConfig(k=50000),
    "vlad-best": VLADConfig(k=4096),
    "fisher-best": FisherConfig(k=256),
    # --- neural codes -----------------------------------------------------------------
    "neural-codes-vgg16": NeuralCodesConfig(backbone="vgg16", dim=None),
    # --- off-the-shelf CNN: the reference's own configuration -------------------------
    # Caffe weights, multi-scale, whitened, and VGG without its trailing max-pool --
    # matching Radenovic et al. rather than torchvision's defaults. See README.
    "gem-r101": PooledConfig(backbone="resnet101", p=3.0, weights="caffe", scales=MULTI_SCALE, dim=2048, whiten=True),
    "gem-vgg16": PooledConfig(
        backbone="vgg16", p=3.0, weights="caffe", last_pool=False, scales=MULTI_SCALE, dim=512, whiten=True
    ),
    "rmac-r101": RMACConfig(backbone="resnet101", levels=3, weights="caffe", scales=MULTI_SCALE, dim=2048, whiten=True),
    # --- fine-tuned checkpoints -------------------------------------------------------
    # `p` and the whitening both come out of the checkpoint, so neither is set here.
    "gem-ft-r101": PooledConfig(
        backbone="resnet101",
        p="learned",
        weights="sfm120k",
        scales=MULTI_SCALE,
        whiten=True,
        whiten_source="learned",
    ),
    "gem-ft-vgg16": PooledConfig(
        backbone="vgg16",
        p="learned",
        weights="sfm120k",
        last_pool=False,
        scales=MULTI_SCALE,
        whiten=True,
        whiten_source="learned",
    ),
}
"""Every preset, keyed by the name its subcommand takes."""
