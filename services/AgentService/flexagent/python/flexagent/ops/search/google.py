# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
import os
from typing import Any, ClassVar, Dict, List, Optional, Type

import requests
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from flexagent.engine import (
    Operator,
    OperatorInputsSchema,
    OperatorSchema,
    Resource,
    Value,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def google_custom_search_compute(
    query: str, subscription_key: str, cx: str, **kwargs: Any
) -> Dict[str, Any]:
    """
    Perform a Google Custom Search and return the search contexts.

    Parameters
    ----------
    query : str
        The search query string.
    subscription_key : str
        Your Google Custom Search API subscription key.
    cx : str
        The Custom Search Engine ID to scope this search.
    timeout : int, optional
        The request timeout in seconds (default is 5).

    Returns
    -------
    dict
        A dictionary containing the search results with keys:
        - 'contexts': List of search results with 'name', 'url', and 'snippet'.

    Raises
    ------
    ValueError
        If there's an error during the search.
    """
    if subscription_key == "" or cx == "":
        raise ValueError("Subscription key and Custom Search Engine ID are required.")
    timeout = kwargs.get("timeout", 10)
    num = kwargs.get("num", 8)  # REFERENCE_COUNT
    search_endpoint = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "key": subscription_key,
        "cx": cx,
        "q": query,
        "num": num,
    }
    logger.info(f"Searching for {query}")

    try:
        response = requests.get(search_endpoint, params=params, timeout=timeout)
        if not response.ok:
            logger.error(
                f"Google Search API Error {response.status_code}: {response.text}"
            )
            raise RequestException(f"Google Search API error: {response.status_code}")

        json_content = response.json()
        contexts = json_content.get("items", [])[:num]

        formatted_contexts = [
            {
                "name": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in contexts
        ]

        return {"contexts": formatted_contexts}

    except RequestException as e:
        logger.error(f"Request exception during Google Custom Search: {e}")
        raise ValueError(f"Error during search: {str(e)}")
    except KeyError as e:
        logger.error(f"Key error processing Google Custom Search response: {e}")
        return {"contexts": []}


class GoogleSearchOpInputsSchema(OperatorInputsSchema):
    query: str = Field(..., description="The search query string.")


class GoogleSearchOpSchema(OperatorSchema):
    name: ClassVar[str] = "google_search"
    description: ClassVar[str] = (
        "Perform a Google Custom Search and return the search contexts."
    )
    parameters: ClassVar[Type[BaseModel]] = GoogleSearchOpInputsSchema


class GoogleSearch(Operator):
    """
    An Operator class for executing Google Custom Search.

    This class extends the Operator class to provide functionality for
    performing Google Custom Searches.

    Parameters
    ----------
    subscription_key : str
        Your Google Custom Search API subscription key.
    cx : str
        The Custom Search Engine ID to scope the search.
    timeout : int, optional
        The request timeout in seconds (default is 5).

    Examples
    --------
    >>> google_search_op = GoogleCustomSearchOperator(
    ...     subscription_key="your_subscription_key",
    ...     cx="your_search_engine_id"
    ... )
    >>> result = google_search_op(query="OpenAI GPT-4")
    >>> print(result['contexts'])
    [{'name': '...', 'url': '...', 'snippet': '...'}, ...]
    """

    def __init__(self, subscription_key: str = "", cx: str = "", timeout: int = 10):
        """
        Initializes the GoogleCustomSearchOperator with necessary credentials.

        Parameters
        ----------
        subscription_key : str
            Your Google Custom Search API subscription key.
        cx : str
            The Custom Search Engine ID to scope the search.
        timeout : int, optional
            The request timeout in seconds (default is 10).
        """
        compute_func = google_custom_search_compute
        super().__init__(compute_func)
        if subscription_key == "":
            subscription_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY", "")
        if cx == "":
            cx = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID", "")
        self.subscription_key = subscription_key
        self.cx = cx
        self.timeout = timeout

    def __call__(
        self, query: str, resources: Optional[List[Resource]] = None, **kwargs: Any
    ) -> Value:
        """
        Executes the Google Custom Search with the given query.

        Parameters
        ----------
        query : str
            The search query string.
        resources : Optional[List[Resource]], optional
            A list of Resource objects that this operator depends on.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Value
            The search result containing contexts.
        """
        return super().__call__(
            query=query,
            subscription_key=self.subscription_key,
            cx=self.cx,
            timeout=self.timeout,
            resources=resources,
            **kwargs,
        )

    @staticmethod
    def get_function_schema() -> Dict[str, Any]:
        return GoogleSearchOpSchema.get_function_schema()
