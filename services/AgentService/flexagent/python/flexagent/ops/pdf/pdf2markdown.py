# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Any, ClassVar, Dict, Type

from pydantic import BaseModel, Field

from flexagent.engine import Operator, OperatorInputsSchema, OperatorSchema


def compute(pdf_uri: str, **kwargs: Any) -> str:
    """
    Convert a PDF file to Markdown.

    :param pdf_uri: The URI of the PDF file to convert.
    :param kwargs: Additional keyword arguments.
    :return: The converted Markdown content as a string.
    """
    try:
        from docling.document_converter import DocumentConverter  # Lazy import
    except ImportError:
        raise ImportError(
            "docling is not installed. Please install it by `pip install flexagent[all]`"
        )
    converter = DocumentConverter()
    result = converter.convert_single(pdf_uri)
    return result.render_as_markdown()


class Pdf2MarkdownOpInputsSchema(OperatorInputsSchema):
    pdf_uri: str = Field(..., description="The URI of the PDF file to convert.")


class Pdf2MarkdownOpSchema(OperatorSchema):
    name: ClassVar[str] = "pdf2markdown"
    description: ClassVar[str] = "Convert a PDF file to Markdown."
    parameters: ClassVar[Type[BaseModel]] = Pdf2MarkdownOpInputsSchema


class Pdf2Markdown(Operator):
    """
    An Operator class for converting PDF to Markdown.

    .. warning::
       This operator should be run in a Docker container for safety.
    """

    def __init__(self) -> None:
        """
        Initialize the Pdf2Markdown operator.

        :param resource_dependencies: Optional list of resource dependencies.
        """
        super().__init__(compute)

    def __call__(self, pdf_uri: str, **kwargs: Any) -> str:
        """
        Convert a PDF file to Markdown.

        :param pdf_uri: The URI of the PDF file to convert.
        :param kwargs: Additional keyword arguments.
        :return: The converted Markdown content as a string.
        """
        return super().__call__(pdf_uri, **kwargs)

    @staticmethod
    def get_function_schema() -> Dict[str, Any]:
        return Pdf2MarkdownOpSchema.get_function_schema()
