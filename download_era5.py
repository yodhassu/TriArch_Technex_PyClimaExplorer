"""
PyClimaExplorer — ERA5 Data Downloader  (TECHNEX'26 · IIT BHU)

Uses the NEW Copernicus CDS API (upgraded 2024).
  - URL : https://cds.climate.copernicus.eu/api   (no /v2)
  - KEY : Personal Access Token only — NO  uid:  prefix

Run once to populate data/ then start the app normally.
"""

import os, sys

# ── Load .env ────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Check cdsapi is installed ────────────────────────────────────
try:
    import cdsapi
except ImportError:
    print("ERROR: cdsapi not installed.")
    print("Run:   pip install cdsapi --upgrade")
    sys.exit(1)

# ── Read credentials ─────────────────────────────────────────────
CDS_KEY = os.environ.get("CDS_API_KEY", "")
CDS_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")

if not CDS_KEY or "xxxx" in CDS_KEY:
    print()
    print("  ERROR: CDS_API_KEY not set correctly in .env")
    print("  ──────────────────────────────────────────────────────")
    print("  The NEW CDS API (2024) uses a Personal Access Token:")
    print()
    print("  1. Login at  https://cds.climate.copernicus.eu/")
    print("  2. Click your name (top-right) → 'Your profile'")
    print("  3. Scroll to the 'API key' section")
    print("  4. Copy the UUID token shown there")
    print("  5. In .env set:")
    print("       CDS_API_URL=https://cds.climate.copernicus.eu/api")
    print("       CDS_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    print("     (just the token — no  uid:  prefix)")
    print()
    sys.exit(1)

# Strip any accidental  uid:  prefix the user may have pasted
if ":" in CDS_KEY:
    CDS_KEY = CDS_KEY.split(":", 1)[1].strip()
    print("  ⚠  Stripped legacy  uid:  prefix from CDS_API_KEY automatically.")

print(f"  ✓  CDS credentials loaded  (URL: {CDS_URL})")
print(f"  ✓  Key (first 8 chars): {CDS_KEY[:8]}...")

# Instantiate client with new API
c = cdsapi.Client(url=CDS_URL, key=CDS_KEY)

# ── Data dir ─────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Download spec ─────────────────────────────────────────────────
# 1° grid keeps each file ≈ 50 MB — change to 0.25 for higher res (×16 larger)
GRID    = [1.0, 1.0]
YEARS   = [str(y) for y in range(1950, 2024)]
MONTHS  = [f"{m:02d}" for m in range(1, 13)]

# New dataset name on CDS (2024 API)
DATASET = "reanalysis-era5-single-levels-monthly-means"

DOWNLOADS = [
    {
        "name":     "temperature",
        "file":     "era5_temperature_monthly.nc",
        "variable": ["2m_temperature"],
    },
    {
        "name":     "precipitation",
        "file":     "era5_precipitation_monthly.nc",
        "variable": ["total_precipitation"],
    },
    {
        "name":     "wind_speed",
        "file":     "era5_wind_monthly.nc",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
    },
    {
        "name":     "solar_radiation",
        "file":     "era5_solar_radiation_monthly.nc",
        "variable": ["surface_solar_radiation_downwards"],
    },
    {
        "name":     "soil_moisture",
        "file":     "era5_soil_moisture_monthly.nc",
        "variable": ["volumetric_soil_water_layer_1"],
    },
    {
        "name":     "snowfall",
        "file":     "era5_snowfall_monthly.nc",
        "variable": ["snowfall"],
    },
]

# ── Download loop ─────────────────────────────────────────────────
ok, skip, fail = 0, 0, 0

for item in DOWNLOADS:
    out_path = os.path.join(DATA_DIR, item["file"])

    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[SKIP] {item['file']} already exists ({size_mb:.0f} MB)")
        skip += 1
        continue

    print(f"\n[DOWNLOAD] {item['name']}")
    print(f"  File     : {item['file']}")
    print(f"  Variables: {item['variable']}")
    print(f"  Years    : {YEARS[0]} – {YEARS[-1]}  ({len(YEARS)} years × 12 months)")
    print(f"  Grid     : {GRID[0]}° × {GRID[1]}°")
    print(f"  This will take a few minutes — CDS queues the job server-side...")

    try:
        c.retrieve(
            DATASET,
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable":     item["variable"],
                "year":         YEARS,
                "month":        MONTHS,
                "time":         "00:00",
                "data_format":  "netcdf",          # new key name in 2024 API
                "download_format": "unarchived",   # get .nc directly, not zipped
                "grid":         GRID,
            },
            out_path,
        )
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  [OK] Saved → {item['file']}  ({size_mb:.1f} MB)")
        ok += 1

    except Exception as e:
        print(f"  [ERROR] {item['name']} failed: {e}")
        # Clean up partial file if it exists
        if os.path.exists(out_path):
            os.remove(out_path)
        fail += 1

# ── Summary ───────────────────────────────────────────────────────
print()
print("═" * 52)
print(f"  Downloaded : {ok}   Skipped : {skip}   Failed : {fail}")
if ok + skip == len(DOWNLOADS):
    print()
    print("  ✅  All ERA5 files ready!")
    print("  Run:  python app.py")
else:
    print()
    print("  ⚠  Some downloads failed — check errors above.")
    print("  Common fixes:")
    print("    • Make sure CDS_API_KEY is your new Personal Access Token")
    print("    • Upgrade cdsapi:  pip install cdsapi --upgrade")
    print("    • Check CDS service status: https://cds.climate.copernicus.eu/")
print("═" * 52)
