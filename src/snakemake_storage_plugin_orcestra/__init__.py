from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

# Raise errors that will not be handled within this plugin but thrown upwards to
# Snakemake and the user as WorkflowError.
from snakemake_interface_common.exceptions import WorkflowError  # noqa: F401
from snakemake_interface_storage_plugins.io import IOCacheStorageInterface
from snakemake_interface_storage_plugins.settings import (
    StorageProviderSettingsBase,
)
from snakemake_interface_storage_plugins.storage_object import (
    StorageObjectGlob,
    StorageObjectRead,
    StorageObjectWrite,
    retry_decorator,
)
from snakemake_interface_storage_plugins.storage_provider import (  # noqa: F401
    ExampleQuery,
    Operation,
    QueryType,
    StorageProviderBase,
    StorageQueryValidationResult,
)
from snakemake_storage_plugin_orcestra.orcestra_helper import manager, similar_names
from urllib import parse


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

    def __post_init__(self):
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
                query_type=QueryType.INPUT,
            )
        ]

    def rate_limiter_key(self, query: str, operation: Operation) -> Any:
        """Return a key for identifying a rate limiter given a query and an operation.
        Notes
        -----
        Unused in orcestra-downloader
        """
        ...

    def default_max_requests_per_second(self) -> float:
        """Return the default maximum number of requests per second for this storage
        provider.
        Notes
        -----
        Unused in orcestra-downloader
        """
        ...

    def use_rate_limiter(self) -> bool:
        """Return False if no rate limiting is needed for this provider.
        Notes
        -----
        Unused in orcestra-downloader
        """
        ...

    @classmethod
    def is_valid_query(cls, query: str) -> StorageQueryValidationResult:
        """Return whether the given query is valid for this storage provider."""
        # Ensure that also queries containing wildcards (e.g. {sample}) are accepted
        # and considered valid. The wildcards will be resolved before the storage
        # object is actually used.
        datatypes = list(manager.names())
        errormsg = ""
        try:
            parsed_query = parse.urlparse(query)
        except Exception as e:
            errormsg = (f"cannot be parsed as URL ({e})",)
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
            elif not parsed_query.path or (parsed_query.split("/") != 2):
                errormsg = f"Invalid path in query '{query}'."
                errormsg += (
                    "Format should follow"
                    f" 'orcestra://<datatype>/<dataset_name>' but got '{parsed_query}'."
                )

        if errormsg:
            return StorageQueryValidationResult(query, False, errormsg)

        dataset_names = manager.registry.get_manager(parsed_query.netloc).names()
        if parsed_query.path not in dataset_names:
            maybe_ds_names = similar_names(parsed_query.path, dataset_names)
            errormsg = (
                f"Dataset '{parsed_query.path}' not found in '{parsed_query.netloc}'."
                f"Did you mean one of {maybe_ds_names}?"
            )
            return StorageQueryValidationResult(False, errormsg)

        return StorageQueryValidationResult(True, "")


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

    def __post_init__(self):
        # This is optional and can be removed if not needed.
        # Alternatively, you can e.g. prepare a connection to your storage backend here.
        # and set additional attributes.
        pass

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
        return f"{parsed.netloc}{parsed.path}"

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
        # return True if the object exists
        ...

    @retry_decorator
    def mtime(self) -> float:
        # return the modification time
        ...

    @retry_decorator
    def size(self) -> int:
        # return the size in bytes
        ...

    @retry_decorator
    def retrieve_object(self) -> None:
        # Ensure that the object is accessible locally under self.local_path()
        ...

    # The following to methods are only required if the class inherits from
    # StorageObjectReadWrite.

    @retry_decorator
    def store_object(self) -> None:
        # Ensure that the object is stored at the location specified by
        # self.local_path().
        ...

    @retry_decorator
    def remove(self) -> None:
        # Remove the object from the storage.
        ...

    # The following to methods are only required if the class inherits from
    # StorageObjectGlob.

    @retry_decorator
    def list_candidate_matches(self) -> Iterable[str]:
        """Return a list of candidate matches in the storage for the query."""
        # This is used by glob_wildcards() to find matches for wildcards in the query.
        # The method has to return concretized queries without any remaining wildcards.
        # Use snakemake_executor_plugins.io.get_constant_prefix(self.query) to get the
        # prefix of the query before the first wildcard.
        ...
