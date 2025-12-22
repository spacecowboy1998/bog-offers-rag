import json
from datetime import date
from pathlib import Path

from src.data_collection.client_offers import OffersApiConfig, fetch_all_offers
from src.data_collection.models import ProcessedOffer

def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _build_taxonomy(processed_offers: list[dict]) -> dict:
    """
    Build taxonomy dynamically from processed offers.
    """

    cities = set()
    categories = set()
    segment_types = set()
    product_codes = set()

    for o in processed_offers:
        for c in (o.get("cityNames") or []):
            if str(c).strip():
                cities.add(str(c).strip())

        cat = (o.get("categoryDesc") or "").strip()
        if cat:
            categories.add(cat)

        for s in (o.get("segmentTypes") or []):
            if str(s).strip():
                segment_types.add(str(s).strip())

        for p in (o.get("productCodes") or []):
            if str(p).strip():
                product_codes.add(str(p).strip())

    return {
        "cities": sorted(cities),
        "categories": sorted(categories),
        "segmentTypes": sorted(segment_types),
        "productCodes": sorted(product_codes),
    }




def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)

    stamp = date.today().isoformat()
    raw_path = raw_dir / f"offers_{stamp}.json"
    processed_path = processed_dir / f"offers_{stamp}.json"
    taxonomy_path = processed_dir / "taxonomy.json"

    config = OffersApiConfig()
    offers_raw = fetch_all_offers(config)

    # Save raw
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(offers_raw, f, ensure_ascii=False, indent=2)

    # Process + keep aliases
    processed_offers = []
    skipped = 0

    for offer in offers_raw:
        try:
            processed = ProcessedOffer.model_validate(offer)
            processed_offers.append(processed.model_dump(mode="json", by_alias=True))
        except Exception:
            skipped += 1

    with processed_path.open("w", encoding="utf-8") as f:
        json.dump(processed_offers, f, ensure_ascii=False, indent=2)

    # Build taxonomy from processed offers
    taxonomy = _build_taxonomy(processed_offers)
    with taxonomy_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print(f"Saved raw:       {raw_path} ({len(offers_raw)} offers)")
    print(f"Saved processed: {processed_path} ({len(processed_offers)} offers, skipped {skipped})")
    print(f"Saved taxonomy:  {taxonomy_path} (cities={len(taxonomy['cities'])}, categories={len(taxonomy['categories'])})")


if __name__ == "__main__":
    main()
