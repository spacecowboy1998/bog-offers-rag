import re
from decimal import Decimal
from datetime import datetime
from typing import Annotated, Any, List, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, BeforeValidator

# Parsing Functions
def parse_semicolon_list(v: Any) -> List[str]:
    """Splits 'A; B' or 'A, B' into ['A', 'B']. Handles single strings and lists."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [t.strip() for t in re.split(r'[;,]', v) if t.strip()]
    return []

def parse_int_list(v: Any) -> List[int]:
    """Parses a list of values into integers, ignoring non-digits."""

    tokens = parse_semicolon_list(v)
    return [int(x) for x in tokens if x.isdigit()]

def clean_html(v: Any) -> str:
    """Strip HTML tags from string."""
    if not v or not isinstance(v, str):
        return ""
    return BeautifulSoup(v, "html.parser").get_text(separator=" ", strip=True)

def parse_timestamp(v: Any) -> datetime:
    """Smart timestamp parser (detects ms vs seconds)."""
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 3e11 else v)
    return v



#  Custom Types for Validation
StringList = Annotated[List[str], BeforeValidator(parse_semicolon_list)]

IntList = Annotated[List[int], BeforeValidator(parse_int_list)]

CleanString = Annotated[str, BeforeValidator(clean_html)]

SmartDate = Annotated[datetime, BeforeValidator(parse_timestamp)]


# Clean modle for offers
class ProcessedOffer(BaseModel):
    # Identifiers
    campaign_id: int = Field(alias="campaignId")

    # Dates
    start_date: SmartDate = Field(alias="startDate")
    end_date:   SmartDate = Field(alias="endDate")
    days_left:  int       = Field(alias="daysLeft")

    # Categories
    category_id:   int = Field(alias="categoryId")
    category_desc: str = Field(alias="categoryDesc")

    # Characteristics
    city_ids:      IntList    = Field(default_factory=list, alias="cityIds")
    city_names:    StringList = Field(default_factory=list, alias="cityNames")
    brand_names:   StringList = Field(default_factory=list, alias="brandNames")
    product_codes: StringList = Field(default_factory=list, alias="productCodes")
    segment_types: StringList = Field(default_factory=list, alias="segmentTypes")

    # Content
    title: str
    short_desc: Optional[str] = Field(default=None, alias="shortDesc")
    long_desc:  CleanString   = Field(alias="longDesc") # Automatically cleans HTML

    # Benefits
    benefit_name:  Optional[str]     = Field(default=None, alias="benefitName")
    benefit_text:  Optional[str]     = Field(default=None, alias="benefText")