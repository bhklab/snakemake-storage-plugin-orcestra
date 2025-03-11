import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional
from urllib import parse

from orcestradownloader.dataset_config import DATASET_CONFIG
from orcestradownloader.logging_config import logger as orcestra_logger
from orcestradownloader.managers import (
    REGISTRY,
    DatasetManager,
    UnifiedDataManager,
)
from orcestradownloader.models.base import BaseModel

# Raise errors that will not be handled within this plugin but thrown upwards to
# Snakemake and the user as WorkflowError.
from snakemake_interface_common.exceptions import WorkflowError  # noqa: F401
from snakemake_interface_storage_plugins.io import IOCacheStorageInterface
from snakemake_interface_storage_plugins.settings import (
    StorageProviderSettingsBase,
)
from snakemake_interface_storage_plugins.storage_object import (
    StorageObjectRead,
    retry_decorator,
)
from snakemake_interface_storage_plugins.storage_provider import (  # noqa: F401
    ExampleQuery,
    Operation,
    QueryType,
    StorageProviderBase,
    StorageQueryValidationResult,
)

if TYPE_CHECKING:
    from datetime import datetime


# Register all dataset managers automatically
for name, config in DATASET_CONFIG.items():
    manager = DatasetManager(
        url=config.url,
        cache_file=config.cache_file,
        dataset_type=config.dataset_type,
    )
    REGISTRY.register(name, manager)

unified_manager = UnifiedDataManager(REGISTRY, force=True)


orcestra_logger.setLevel("WARNING")

for handler in orcestra_logger.handlers[:]:
    orcestra_logger.removeHandler(handler)
    handler.close()


@dataclass
class StorageProviderSettings(StorageProviderSettingsBase):
    pass


# Required:
# Implementation of your storage provider
# This class can be empty as the one below.
# You can however use it to store global information or maintain e.g. a connection
# pool.
class StorageProvider(StorageProviderBase):
    # For compatibility with future changes, you should not overwrite the __init__
    # method. Instead, use __post_init__ to set additional attributes and initialize
    # futher stuff.

    def __post_init__(self) -> None:
        # This is optional and can be removed if not needed.
        # Alternatively, you can e.g. prepare a connection to your storage backend here.
        # and set additional attributes.
        pass

    @classmethod
    def example_queries(cls) -> List[ExampleQuery]:
        """Return an example queries with description for this storage provider (at
        least one)."""
        return [
            ExampleQuery(
                query="orcestra://pharmacosets/CCLE_2015",
                description="Download the CCLE 2015 dataset.",
                type=QueryType.INPUT,
            )
        ]

    def rate_limiter_key(self, query: str, operation: Operation) -> Any:  # noqa: ANN401
        """Return a key for identifying a rate limiter given a query and an operation.
        Notes
        -----
        Unused in orcestra-downloader
        """
        return None

    def default_max_requests_per_second(self) -> float:
        """Return the default maximum number of requests per second for this storage
        provider.
        Notes
        -----
        Unused in orcestra-downloader
        """
        return 0.0

    def use_rate_limiter(self) -> bool:
        """Return False if no rate limiting is needed for this provider.
        Notes
        -----
        Unused in orcestra-downloader
        """
        return False

    @classmethod
    def is_valid_query(cls, query: str) -> StorageQueryValidationResult:
        """Return whether the given query is valid for this storage provider."""
        # Ensure that also queries containing wildcards (e.g. {sample}) are accepted
        # and considered valid. The wildcards will be resolved before the storage
        # object is actually used.
        datatypes = list(unified_manager.names())
        errormsg = ""
        try:
            parsed_query = parse.urlparse(query)
        except Exception as e:
            errormsg = f"cannot be parsed as URL ({e})"
        else:
            if parsed_query.scheme != "orcestra":
                errormsg = (
                    f"Invalid scheme in query '{query}'."
                    f"{parsed_query.scheme} should be 'orcestra'."
                )
            elif parsed_query.netloc not in datatypes:
                errormsg = (
                    f"Invalid netloc in query '{query}'."
                    f"{parsed_query.netloc} should be one of {datatypes}."
                )
            elif not parsed_query.path:
                # remove the slash at the beginning
                dataset_name = parsed_query.path[1:]
                # check if there are still slashes in the path
                if "/" in dataset_name:
                    errormsg = (
                        f"Invalid path in query '{query}'. "
                        f"Format should follow"
                        " 'orcestra://<datatype>/<dataset_name>' but got '{parsed_query}'."
                    )

        if errormsg:
            orcestra_logger.error(errormsg)
            return StorageQueryValidationResult(query, False, errormsg)

        return StorageQueryValidationResult(query, True, "")


# Required:
# Implementation of storage object. If certain methods cannot be supported by your
# storage (e.g. because it is read-only see
# snakemake-storage-http for comparison), remove the corresponding base classes
# from the list of inherited items.
class StorageObject(StorageObjectRead):
    # following attributes are inherited from StorageObjectRead:
    # query = query
    # keep_local = keep_local
    # retrieve = retrieve
    # provider = provider
    # _overwrite_local_path = None

    dataset_type: str = field(init=False)
    dataset_name: str = field(init=False)
    manager: DatasetManager = field(init=False)
    dataset_metadata: BaseModel | None = field(init=False)

    def __post_init__(self) -> None:
        # This is optional and can be removed if not needed.
        # Alternatively, you can e.g. prepare a connection to your storage backend here.
        # and set additional attributes.
        orcestra_logger.debug(f"StorageObject for query {self.query} created.")
        orcestra_logger.debug(
            f"Arguments: {self.keep_local=}, {self.retrieve=}, {self.provider=}, {self._overwrite_local_path=}"
        )
        parsed = parse.urlparse(self.query)
        self.dataset_type = parsed.netloc
        self.dataset_name = parsed.path.split("/")[1]

        orcestra_logger.debug(
            f"Dataset type: {self.dataset_type} and name: {self.dataset_name}"
        )

        # initialize manager for this datatype
        self.manager = unified_manager.registry.get_manager(self.dataset_type)

        # use unified manager to fetch info
        try:
            # Get the current event loop or create one if it doesn't exist
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new future task and run it directly
                asyncio.create_task(
                    unified_manager.fetch_by_name(
                        self.dataset_type, force=True
                    )
                )
            else:
                # If no loop is running, run the coroutine directly
                loop.run_until_complete(
                    unified_manager.fetch_by_name(
                        self.dataset_type, force=True
                    )
                )
        except RuntimeError:
            # Fallback to creating a new loop if needed
            asyncio.run(
                unified_manager.fetch_by_name(self.dataset_type, force=True)
            )

        try:
            self.dataset_metadata = self.manager[self.dataset_name]
        except ValueError:
            self.dataset_metadata = None

    async def inventory(self, cache: IOCacheStorageInterface) -> None:
        """From this file, try to find as much existence and modification date
        information as possible. Only retrieve that information that comes for free
        given the current object.
        """
        # This is optional and can be left as is

        # If this is implemented in a storage object, results have to be stored in
        # the given IOCache object, using self.cache_key() as key.
        # Optionally, this can take a custom local suffix, needed e.g. when you want
        # to cache more items than the current query: self.cache_key(local_suffix=...)
        pass

    def get_inventory_parent(self) -> Optional[str]:
        """Return the parent directory of this object."""
        # this is optional and can be left as is
        return None

    def local_suffix(self) -> str:
        """Return a unique suffix for the local path, determined from self.query."""
        parsed = parse.urlparse(self.query)
        return f"{parsed.netloc}{parsed.path}.RDS"

    def cleanup(self) -> None:
        """Perform local cleanup of any remainders of the storage object."""
        # self.local_path() should not be removed, as this is taken care of by
        # Snakemake.
        ...

    # Fallible methods should implement some retry logic.
    # The easiest way to do this (but not the only one) is to use the retry_decorator
    # provided by snakemake-interface-storage-plugins.
    @retry_decorator
    def exists(self) -> bool:
        if self.dataset_metadata:
            return True
        from difflib import get_close_matches

        dataset_names = self.manager.names()

        if self.dataset_name not in dataset_names:
            errmsg = (
                f"Dataset {self.dataset_name} not found in {self.dataset_type}."
                f"Did you mean one of {get_close_matches(self.dataset_name, dataset_names)}?"
            )
            orcestra_logger.error(errmsg)
        return False

    @retry_decorator
    def mtime(self) -> float:
        # return the modification time
        if self.dataset_metadata is None:
            # return infinity if no date is available
            return float("-inf")

        created_date: datetime | None = self.dataset_metadata.date_created

        if created_date is None:
            # return infinity if no date is available
            return float("-inf")
        return float(created_date.timestamp())

    @retry_decorator
    def size(self) -> int:
        # return the size in bytes
        return 0

    @retry_decorator
    def retrieve_object(self) -> None:
        # Ensure that the object is accessible locally under self.local_path()
        directory_path = Path(self.local_path()).parent
        if not directory_path.exists():
            directory_path.mkdir(parents=True)
        from rich.progress import (
            Progress,
        )

        if self.dataset_metadata is None:
            errmsg = f"Dataset {self.dataset_name} not found in {self.dataset_type}."
            orcestra_logger.error(errmsg)
            raise WorkflowError(errmsg)

        download_url = self.dataset_metadata.download_link

        if download_url is None:
            errmsg = f"Download URL for dataset {self.dataset_name} not found in {self.dataset_type}."
            orcestra_logger.error(errmsg)
            raise WorkflowError(errmsg)

        import requests

        with Progress() as progress:
            task = progress.add_task(
                f"Downloading {self.dataset_name} from {self.dataset_type}",
                total=100,
            )
            temp_file = Path(f"{self.local_path()}.temp")
            # Download the dataset to the local path
            # with open(temp_file, "wb") as f:
            with temp_file.open("wb") as f:
                response = requests.get(download_url, stream=True)
                # get total size based on header content-length
                total_size = int(response.headers.get("content-length", 0))

                if total_size == 0:
                    orcestra_logger.warning(
                        f"Could not determine total size of download for {self.dataset_name}."
                    )
                task = progress.add_task(
                    f"Downloading {self.dataset_name} from {self.dataset_type}",
                    total=total_size,
                )
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

            # rename the temp file to the final file
            Path(temp_file).rename(self.local_path())
            progress.update(task, completed=100)
            progress.stop()

        orcestra_logger.debug(f"Downloaded dataset to {self.local_path()}")

    # The following to methods are only required if the class inherits from
    # StorageObjectReadWrite.

    # @retry_decorator
    # def store_object(self) -> None:
    #     # Ensure that the object is stored at the location specified by
    #     # self.local_path().
    #     ...

    # @retry_decorator
    # def remove(self) -> None:
    #     # Remove the object from the storage.
    #     ...

    # The following to methods are only required if the class inherits from
    # StorageObjectGlob.

    # @retry_decorator
    # def list_candidate_matches(self) -> Iterable[str]:
    #     """Return a list of candidate matches in the storage for the query."""
    #     # This is used by glob_wildcards() to find matches for wildcards in the query.
    #     # The method has to return concretized queries without any remaining wildcards.
    #     # Use snakemake_executor_plugins.io.get_constant_prefix(self.query) to get the
    #     # prefix of the query before the first wildcard.
    #     ...
