from orcestradownloader.managers import (
    REGISTRY,
    DatasetRegistry,
    DatasetManager,
    UnifiedDataManager,
)

from orcestradownloader.dataset_config import DATASET_CONFIG, DatasetConfig
import difflib


def similar_names(
    query: str, names: list[str], n: int = 3, cutoff: float = 0.5
) -> list[str]:
    return difflib.get_close_matches(query, names, n=n, cutoff=cutoff)


# Register all dataset managers automatically
for name, config in DATASET_CONFIG.items():
    manager = DatasetManager(
        url=config.url, cache_file=config.cache_file, dataset_type=config.dataset_type
    )
    REGISTRY.register(name, manager)

manager = UnifiedDataManager(REGISTRY)
manager.hydrate_cache()
