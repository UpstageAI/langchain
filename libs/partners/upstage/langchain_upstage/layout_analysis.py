import os
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from .parsers import LayoutAnalysisParser

LIMIT_OF_PAGE_REQUEST = 10

OutputType = Literal["text", "html"]
SplitType = Literal["none", "element", "page"]


def validate_api_key(api_key: str) -> None:
    """
    Validates the provided API key.

    Args:
        api_key (str): The API key to be validated.

    Raises:
        ValueError: If the API key is empty or None.

    Returns:
        None
    """
    if not api_key:
        raise ValueError("API Key is required for Upstage Document Loader")


def validate_file_path(file_path: str) -> None:
    """
    Validates if a file exists at the given file path.

    Args:
        file_path (str): The path to the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class LayoutAnalysis:
    def __init__(
        self,
        file_path: Union[str, Path],
        output_type: OutputType = "text",
        split: SplitType = "none",
        api_key: str = None,
    ):
        """
        Initializes an instance of the Upstage document loader.

        Args:
            file_path (Union[str, Path]): The path to the file to be loaded.
            output_type (OutputType, optional): The desired output type.
                                                Defaults to "text".
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none".
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
        """
        self.file_path = file_path
        self.output_type = output_type
        self.split = split
        self.file_name = os.path.basename(file_path)
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY"
        )

        validate_file_path(self.file_path)
        validate_api_key(self.api_key)

    def load(self) -> List[Document]:
        """
        oads and parses the document using the LayoutAnalysisParser.

        Returns:
            A list of Document objects representing the parsed layout analysis.
        """
        blob = Blob.from_path(self.file_path)

        parser = LayoutAnalysisParser(
            self.api_key, split=self.split, output_type=self.output_type
        )
        return list(parser.lazy_parse(blob, split_pages=LIMIT_OF_PAGE_REQUEST))

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads and parses the document using the LayoutAnalysisParser.

        Returns:
            An iterator of Document objects representing the parsed layout analysis.
        """

        blob = Blob.from_path(self.file_path)

        parser = LayoutAnalysisParser(
            self.api_key, split=self.split, output_type=self.output_type
        )
        yield from parser.lazy_parse(blob)
