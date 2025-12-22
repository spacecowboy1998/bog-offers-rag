from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import requests


@dataclass(frozen=True)
class OffersApiConfig:
    base_url: str = "https://bankofgeorgia.ge/api/bog-b/offers-hub/get-offers"
    page_size: int = 100
    timeout_s: float = 15.0


def fetch_all_offers(config: OffersApiConfig) -> List[Dict[str, Any]]:
    """
    Fetch all offers from the BoG offers hub API via pagination.

    Returns:
    List of raw offer dicts from the API.
    """
    session = requests.Session()
    all_offers: List[Dict[str, Any]] = []
    page_number = 0

    while True:
        params = {
            "pageInfo": "false",
            "pageNumber": page_number,
            "pageSize": config.page_size,
        }

        # Starting Session with API
        resp = session.post(config.base_url, params=params, timeout=config.timeout_s)
        resp.raise_for_status()

        payload = resp.json()
        current_offers = payload.get("result", {}).get("offers", [])

        if not isinstance(current_offers, list):
            raise ValueError("API response: result.offers is not a list")

        all_offers.extend(current_offers)
        fetched = len(current_offers)
        print(f"Successfully fetched {fetched} offers from page {page_number}")

        # Stop condition: if we fetched fewer items than the page size, it's the last page
        if fetched < config.page_size:
            return all_offers

        page_number += 1