"""
sku_store.py
------------
In-memory SKU metadata store for the Supply Chain Agent.
- Pydantic V2 compliant
- Bare @tool decorator (description lives in docstring)
- Robust fallback mock dictionary when CSV is absent or malformed
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional LangChain import
# ---------------------------------------------------------------------------
try:
    from langchain_core.tools import tool
except ImportError:
    def tool(fn):  # type: ignore[misc]
        return fn

# ---------------------------------------------------------------------------
# 1.  Pydantic V2 input schema
# ---------------------------------------------------------------------------

class SKUProfileInput(BaseModel):
    """Input schema for the get_sku_profile tool."""

    sku_id: str = Field(..., description="Unique SKU identifier (e.g. 'HOBBIES_1_CA').")


# ---------------------------------------------------------------------------
# 2.  Internal database loader
# ---------------------------------------------------------------------------

_FEATURES_CSV_PATH = Path(__file__).parent / "Dataset" / "m5_demand_features.csv"

# Fallback mock catalogue — representative demand archetypes from the M5 dataset.
_MOCK_SKU_DB: Dict[str, Dict[str, float]] = {
    "SKU_FAST_MOVING": {"adi": 1.1,  "cv2": 0.2},
    "SKU_LUMPY":       {"adi": 4.5,  "cv2": 1.5},
    "SKU_SEASONAL":    {"adi": 7.2,  "cv2": 2.3},
    "SKU_TRENDING":    {"adi": 2.8,  "cv2": 0.8},
    "SKU_SLOW":        {"adi": 10.0, "cv2": 3.0},
}

_SKU_DB: Dict[str, Dict[str, float]] = {}


def _load_sku_database() -> Dict[str, Dict[str, float]]:
    """Load ADI and CV2 per SKU into an in-memory dict for O(1) look-ups.

    Tries to read ``Dataset/m5_demand_features.csv``. Falls back to the
    built-in mock catalogue on any I/O or parsing error so the agent
    never crashes at start-up.

    Returns
    -------
    dict
        Mapping ``sku_id -> {"adi": float, "cv2": float}``.
    """
    global _SKU_DB

    if _SKU_DB:
        return _SKU_DB  # already loaded

    if _FEATURES_CSV_PATH.is_file():
        try:
            df = pd.read_csv(_FEATURES_CSV_PATH)

            # Normalise column names (handles mixed-case CSV headers)
            df.columns = [c.strip().lower() for c in df.columns]

            id_col  = next((c for c in df.columns if c in {"id", "sku_id", "item_id"}), None)
            state_col = next((c for c in df.columns if "state" in c), None)
            adi_col = next((c for c in df.columns if "adi" in c), None)
            cv2_col = next((c for c in df.columns if "cv2" in c or "cv_2" in c), None)

            if id_col and adi_col and cv2_col:
                new_db: Dict[str, Dict[str, float]] = {}
                for _, row in df.iterrows():
                    item = str(row[id_col]).strip()
                    adi_val = float(row[adi_col])
                    cv2_val = float(row[cv2_col])
                    profile = {"adi": adi_val, "cv2": cv2_val}

                    if state_col and pd.notna(row.get(state_col)):
                        state = str(row[state_col]).strip()
                        composite = f"{item}_{state}"
                        new_db[composite] = profile
                        new_db[composite.upper()] = profile

                    if item not in new_db:
                        new_db[item] = profile
                        new_db[item.upper()] = profile

                # Also inject mock archetypes so demo SKU names always resolve
                new_db.update(_MOCK_SKU_DB)
                _SKU_DB = new_db
                print(f"[sku_store] Loaded {len(_SKU_DB):,} SKU keys from CSV (+ {len(_MOCK_SKU_DB)} mock archetypes).")
                return _SKU_DB
            else:
                print(
                    f"[sku_store] Warning: expected columns not found "
                    f"(id={id_col}, state={state_col}, adi={adi_col}, cv2={cv2_col}). "
                    "Falling back to mock data."
                )
        except Exception as exc:
            print(f"[sku_store] Warning: failed to parse CSV ({exc}). Falling back to mock data.")

    print("[sku_store] Using built-in mock SKU catalogue.")
    _SKU_DB = dict(_MOCK_SKU_DB)
    return _SKU_DB


# Load once at import time.
_load_sku_database()


# ---------------------------------------------------------------------------
# 3.  LangChain tool
# ---------------------------------------------------------------------------

@tool
def get_sku_profile(sku_id: str) -> Dict[str, float]:
    """Retrieve the structural demand profile (ADI and CV2) for a specific SKU.

    This tool MUST be called first before simulate_policy_outcomes.
    It returns the real demand characteristics stored in the metadata store.
    Never skip this step or guess ADI / CV2 values.

    Parameters
    ----------
    sku_id : str
        The unique product identifier (e.g. 'SKU_FAST_MOVING', 'FOODS_1_001_CA', 'HOBBIES_1_001_CA').

    Returns
    -------
    dict
        ``{"adi": <float>, "cv2": <float>}`` for the requested SKU.
        Raises ValueError if the SKU is not found, prompting the user
        to supply a valid identifier.
    """
    db = _SKU_DB
    clean_id = sku_id.strip()
    if clean_id in db:
        return db[clean_id]
    if clean_id.upper() in db:
        return db[clean_id.upper()]

    available_samples = [k for k in list(db.keys()) if "_" in k][:8]
    return f"Error: SKU '{sku_id}' not found. Examples of valid SKUs: {available_samples}"
        
    return db[sku_id]


__all__ = ["SKUProfileInput", "get_sku_profile", "_load_sku_database"]
