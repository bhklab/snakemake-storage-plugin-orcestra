import difflib

from orcestradownloader.dataset_config import DATASET_CONFIG
from orcestradownloader.managers import (
    REGISTRY,
    DatasetManager,
    UnifiedDataManager,
)


def similar_names(
    query: str, names: list[str], n: int = 3, cutoff: float = 0.5
) -> list[str]:
    return difflib.get_close_matches(query, names, n=n, cutoff=cutoff)


# Register all dataset managers automatically
for name, config in DATASET_CONFIG.items():
    manager = DatasetManager(
        url=config.url,
        cache_file=config.cache_file,
        dataset_type=config.dataset_type,
    )
    REGISTRY.register(name, manager)

unified_manager = UnifiedDataManager(REGISTRY, force=True)
