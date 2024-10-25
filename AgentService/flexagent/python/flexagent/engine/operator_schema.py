# Copyright 2024 NVIDIA Corporation. All rights reserved.

from typing import Any, ClassVar, Dict, get_args, get_origin, Type, Union

from pydantic import BaseModel


class OperatorSchema(BaseModel):
    """
    A base class for defining operator schemas.

    This class provides a structure for defining operator schemas with name, description,
    and parameters. It also includes methods for generating function schemas.

    Attributes:
        name (ClassVar[str]): The name of the operator.
        description (ClassVar[str]): A description of the operator.
        parameters (ClassVar[Type[BaseModel]]): A Pydantic model defining the operator's parameters.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Type[BaseModel]]

    class Config:
        extra = "forbid"

        @classmethod
        def model_modify_json_schema(cls, json_schema):
            """
            Modify the JSON schema by removing the 'title' field.

            Args:
                json_schema (dict): The original JSON schema.

            Returns:
                dict: The modified JSON schema.
            """
            # Remove 'title' from the JSON schema
            json_schema.pop("title", None)
            return json_schema

    @classmethod
    def get_function_schema(cls) -> Dict[str, Any]:
        """
        Generate a function schema based on the operator's parameters.

        This method creates a JSON-compatible schema that describes the operator's
        parameters, including their types and descriptions.

        Returns:
            Dict[str, Any]: A dictionary representing the function schema.
        """
        # Generate the parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        # Access the fields of the parameters model
        for field_name, field_info in cls.parameters.model_fields.items():
            # Get the annotation (type) of the field
            field_type = field_info.annotation
            field_schema = {}

            # Handle typing.Optional
            if get_origin(field_type) is Union:
                field_types = [t for t in get_args(field_type) if t is not type(None)]
                field_type = field_types[0] if field_types else Any
                is_optional = True
            else:
                is_optional = False

            # Map Python types to JSON Schema types
            origin_type = get_origin(field_type)
            if origin_type is list:
                # Handle list types
                item_type = get_args(field_type)[0]
                item_type_name = cls._map_python_type_to_json_type(item_type)
                field_schema = {
                    "type": "array",
                    "items": {"type": item_type_name},
                    "description": field_info.description or "",
                }
            else:
                field_type_name = cls._map_python_type_to_json_type(field_type)
                field_schema = {
                    "type": field_type_name,
                    "description": field_info.description or "",
                }

            parameters_schema["properties"][field_name] = field_schema
            if not is_optional:
                parameters_schema["required"].append(field_name)

        # Construct the final function schema
        function_schema = {
            "name": cls.name,
            "description": cls.description,
            "parameters": parameters_schema,
        }
        return function_schema

    @staticmethod
    def _map_python_type_to_json_type(py_type):
        """
        Map Python types to corresponding JSON schema types.

        Args:
            py_type: The Python type to be mapped.

        Returns:
            str: The corresponding JSON schema type.
        """
        if py_type is str:
            return "string"
        elif py_type is int:
            return "integer"
        elif py_type is float:
            return "number"
        elif py_type is bool:
            return "boolean"
        elif isinstance(py_type, type) and issubclass(py_type, BaseModel):
            return "object"
        elif get_origin(py_type) is list:
            return "array"
        else:
            return "string"  # Default to string for simplicity


class OperatorInputsSchema(BaseModel):
    """
    A base class for defining operator input schemas.

    This class is used to create specific input schemas for operators.
    """

    class Config:
        extra = "forbid"
