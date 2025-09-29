"""
URL parsing and categorization logic
"""

import re
from typing import List
from urllib.parse import urlparse

from .models import ModelContext, ParsedURL, URLCategory


def parse_url(url: str) -> ParsedURL:
    #url parsing and categorizing
    parsed = urlparse(url.strip())

    if "huggingface.co" in parsed.netloc:
        return _parse_huggingface_url(url, parsed)
    elif "github.com" in parsed.netloc:
        return _parse_github_url(url, parsed)
        
    else:
        # for unknown platforms, setting best guess to Dataset
        return ParsedURL(
            url=url,
            category=URLCategory.DATASET,  #defaults into
            name=url.split("/")[-1] or url,
            platform="unknown",
        )


def _parse_huggingface_url(url: str, parsed) -> ParsedURL:
    #parse hugging face urls
    path_parts = [part for part in parsed.path.split("/") if part]

    if not path_parts:
        raise ValueError(f"Invalid Hugging Face URL: {url}")

    #dataset route is /datasets/<owner>/<repo>
    if len(path_parts) >= 2 and path_parts[0] == "datasets":
        category = URLCategory.DATASET
        owner = path_parts[1] if len(path_parts) > 1 else None
        repo = path_parts[2] if len(path_parts) > 2 else None
        #set name as "owner/repo" when both exist
        if owner and repo:
            name = f"{owner}/{repo}"
        else:
            name = repo if repo else (url.split("/")[-1])
    else:
        #otherwise treat as model
        category = URLCategory.MODEL
        owner = path_parts[0] if len(path_parts) > 0 else None
        repo = path_parts[1] if len(path_parts) > 1 else None
        #use only the repo name for models if it exists
        name = repo if repo else (url.split("/")[-1])

    return ParsedURL(
        url=url,
        category=category,
        name=name,
        platform="huggingface",
        owner=owner,
        repo=repo,
    )


def _parse_github_url(url: str, parsed) -> ParsedURL:
    path_parts = [part for part in parsed.path.split("/") if part]

    if len(path_parts) < 2:
        #requires atleast owner/repo
        raise ValueError(f"Invalid GitHub URL: {url}")

    owner = path_parts[0]
    repo = path_parts[1]
    name = f"{owner}/{repo}"

    return ParsedURL(
        url=url,
        category=URLCategory.CODE,
        name=name,
        platform="github",
        owner=owner,
        repo=repo,
    )


def build_model_contexts(urls: List[str]) -> List[ModelContext]:
    """
    Build model contexts by linking datasets and code to models.

    Assumption: datasets and code appear before their associated models.
    """
    parsed_urls = [parse_url(url) for url in urls]
    contexts: List[ModelContext] = []

    #accumulate datasets and code
    pending_datasets: List[ParsedURL] = []
    pending_code: List[ParsedURL] = []

    for parsed_url in parsed_urls:
        if parsed_url.category == URLCategory.DATASET:
            pending_datasets.append(parsed_url)
        elif parsed_url.category == URLCategory.CODE:
            pending_code.append(parsed_url)
        elif parsed_url.category == URLCategory.MODEL:
            #if we hit a model, attach whatever we have
            context = ModelContext(
                model_url=parsed_url,
                datasets=_find_relevant_resources(parsed_url, pending_datasets),
                code_repos=_find_relevant_resources(parsed_url, pending_code),
            )
            contexts.append(context)

            #don't clear pending resources, they might apply to multiple models

    return contexts


def _find_relevant_resources(
    model_url: ParsedURL, resources: List[ParsedURL]
) -> List[ParsedURL]:
    #finding resources based on name similarity
    if not resources:
        return []

    relevant = []
    model_name_parts = set(_extract_name_parts(model_url.name))

    for resource in resources:
        resource_name_parts = set(_extract_name_parts(resource.name))

        #overlap on tokens QR same owner = good link
        if model_name_parts & resource_name_parts or model_url.owner == resource.owner:
            relevant.append(resource)

    #fallback if no matches found
    if not relevant and resources:
        #include the most recent resources then
        relevant = resources[-2:] if len(resources) >= 2 else resources

    return relevant


def _extract_name_parts(name: str) -> List[str]:
    """Extract meaningful parts from a name for comparison."""
    #split on common separators and extract alphabetic parts
    parts = re.split(r"[/_\-\s.]+", name.lower())
    return [part for part in parts if part and part.isalpha() and len(part) > 2]
