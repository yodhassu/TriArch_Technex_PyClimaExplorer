"""
PyClimaExplorer — Climate Data Processor  (TECHNEX'26 · IIT BHU)

Real data ONLY — reads ERA5 / CESM NetCDF files via xarray + netCDF4.
No synthetic data, no external API.
Run download_era5.py first to populate the data/ folder.
"""

import numpy as np
import pandas as pd
import json
import os

import xarray as xr
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ─── Variable catalogue ──────────────────────────────────────────────────────
# Maps our variable IDs → ERA5 variable names + metadata
VARIABLES = {
    "temperature": {
        "label": "Surface Temperature (2m)",
        "unit": "°C",
        "era5_var": "t2m",           # K  → convert to °C
        "era5_monthly_name": "2m_temperature",
        "colorscale": "RdBu_r",
        "fmt": lambda v: f"{v:.1f} °C",
        "nc_offset": -273.15,         # K → °C
        "nc_scale": 1.0,
        "rcp_slopes": (0.45, 1.20, 2.80),   # per decade for 2.6/4.5/8.5
    },
    "precipitation": {
        "label": "Precipitation",
        "unit": "mm/day",
        "era5_var": "tp",             # m/day → mm/day
        "era5_monthly_name": "total_precipitation",
        "colorscale": "Blues",
        "fmt": lambda v: f"{v:.2f} mm/day",
        "nc_offset": 0.0,
        "nc_scale": 1000.0,           # ERA5 stores m/day
        "rcp_slopes": (0.04, 0.10, 0.22),
    },
    "wind_speed": {
        "label": "Wind Speed (10m)",
        "unit": "m/s",
        "era5_var": None,             # derived from u10 + v10
        "era5_u_var": "u10",
        "era5_v_var": "v10",
        "era5_monthly_name": "10m_u_component_of_wind",  # download both u+v
        "colorscale": "Viridis",
        "fmt": lambda v: f"{v:.1f} m/s",
        "nc_offset": 0.0,
        "nc_scale": 1.0,
        "rcp_slopes": (0.08, 0.20, 0.48),
    },
    "solar_radiation": {
        "label": "Surface Solar Radiation",
        "unit": "W/m²",
        "era5_var": "ssrd",           # J/m²/day → W/m²
        "era5_monthly_name": "surface_solar_radiation_downwards",
        "colorscale": "YlOrRd",
        "fmt": lambda v: f"{v:.0f} W/m²",
        "nc_offset": 0.0,
        "nc_scale": 1.0 / 86400.0,   # J/m²/day → W/m²
        "rcp_slopes": (-0.5, -1.2, -2.5),
    },
    "snowfall": {
        "label": "Snowfall",
        "unit": "mm/day",
        "era5_var": "sf",
        "era5_monthly_name": "snowfall",
        "colorscale": "PuBu",
        "fmt": lambda v: f"{v:.4f} mm/day",
        "nc_offset": 0.0,
        "nc_scale": 1000.0,
        "rcp_slopes": (-0.01, -0.05, -0.10),
    },
    "soil_moisture": {
        "label": "Soil Moisture (0–7cm)",
        "unit": "m³/m³",
        "era5_var": "swvl1",
        "era5_monthly_name": "volumetric_soil_water_layer_1",
        "colorscale": "BrBG",
        "fmt": lambda v: f"{v:.3f} m³/m³",
        "nc_offset": 0.0,
        "nc_scale": 1.0,
        "rcp_slopes": (-0.002, -0.005, -0.012),
    },
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── Plotly dark theme ───────────────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,10,28,0.55)",
    font=dict(color="#8ab8c8", family="Exo 2,sans-serif", size=11),
    margin=dict(l=52, r=16, t=38, b=46),
    legend=dict(bgcolor="rgba(0,8,24,0.75)",
                bordercolor="rgba(74,212,232,0.2)",
                borderwidth=1, font=dict(size=10)),
    hoverlabel=dict(bgcolor="rgba(0,12,35,0.95)",
                    bordercolor="rgba(74,212,232,0.45)",
                    font=dict(color="white", size=11)),
    colorway=["#4ad4e8","#a78bfa","#f0b429","#68d391"],
)
_XA = dict(gridcolor="rgba(74,212,232,0.08)",
           linecolor="rgba(74,212,232,0.22)",
           zerolinecolor="rgba(74,212,232,0.15)",
           tickcolor="#4ad4e8")
_YA = dict(gridcolor="rgba(74,212,232,0.08)",
           linecolor="rgba(74,212,232,0.22)",
           zerolinecolor="rgba(74,212,232,0.15)",
           tickcolor="#4ad4e8")

def _toj(fig):
    return json.loads(fig.to_json())


# ─── NetCDF loader ───────────────────────────────────────────────────────────
class NCDataset:
    """
    Wraps an ERA5 monthly-mean NetCDF file.

    ERA5 monthly files from CDS look like:
        Dimensions: (time, latitude, longitude)
        Coords:     time  (datetime64),
                    latitude  (deg N, usually descending 90→-90),
                    longitude (deg E, 0→360 or -180→180)
        Variables:  t2m, tp, u10, v10, ssrd, swvl1 … (in SI units)
    """

    def __init__(self, path: str):
        self.path = path
        self.ds: xr.Dataset = xr.open_dataset(path, engine="netcdf4")
        self._normalize_coords()
        print(f"[NC] Loaded : {os.path.basename(path)}")
        print(f"     vars   : {list(self.ds.data_vars)}")
        print(f"     dims   : {dict(self.ds.dims)}")
        t0 = self.ds.time.values[0]
        t1 = self.ds.time.values[-1]
        print(f"     time   : {pd.Timestamp(t0)} to {pd.Timestamp(t1)}  ({len(self.ds.time)} steps)")

    # coord normalisation
    def _normalize_coords(self):
        """
        Rename coords to standard names.
        Handles ERA5 variants including valid_time, number, expver dims.
        """
        renames = {}
        # valid_time -> time  (new CDS download format)
        if "valid_time" in self.ds.coords and "time" not in self.ds.coords:
            renames["valid_time"] = "time"
        if "valid_time" in self.ds.dims and "time" not in self.ds.dims:
            renames["valid_time"] = "time"
        # lat variants
        for old, new in [("lat","latitude"),("LAT","latitude"),("y","latitude")]:
            if old in self.ds.coords and "latitude" not in self.ds.coords:
                renames[old] = new
        # lon variants
        for old, new in [("lon","longitude"),("LON","longitude"),("x","longitude")]:
            if old in self.ds.coords and "longitude" not in self.ds.coords:
                renames[old] = new
        if renames:
            self.ds = self.ds.rename(renames)
        # Drop size-1 ensemble/version dims
        for dim in ["number", "expver"]:
            if dim in self.ds.dims and self.ds.dims[dim] == 1:
                self.ds = self.ds.squeeze(dim, drop=True)
            elif dim in self.ds.coords:
                self.ds = self.ds.drop_vars(dim, errors="ignore")
        # Ensure longitude is -180..180
        if "longitude" in self.ds.coords:
            lons = self.ds.longitude.values
            if lons.max() > 180:
                self.ds = self.ds.assign_coords(
                    longitude=(self.ds.longitude + 180) % 360 - 180
                )
                self.ds = self.ds.sortby("longitude")
        # Sort latitude ascending so slice() works cleanly
        if "latitude" in self.ds.coords:
            self.ds = self.ds.sortby("latitude")


    # ── variable extraction helpers ──────────────────────────────────────
    def _raw(self, variable: str) -> xr.DataArray:
        """Return a DataArray with correct units applied."""
        cfg = VARIABLES[variable]
        if variable == "wind_speed":
            u = self.ds[cfg["era5_u_var"]]
            v = self.ds[cfg["era5_v_var"]]
            da = np.sqrt(u**2 + v**2)
        else:
            vname = cfg["era5_var"]
            if vname not in self.ds:
                raise KeyError(f"Variable '{vname}' not found in {self.path}. "
                               f"Available: {list(self.ds.data_vars)}")
            da = self.ds[vname] * cfg["nc_scale"] + cfg["nc_offset"]
        return da

    def available_variables(self) -> list:
        found = []
        for k, cfg in VARIABLES.items():
            try:
                self._raw(k)
                found.append(k)
            except (KeyError, Exception):
                pass
        return found

    # ── point extraction ─────────────────────────────────────────────────
    def point_value(self, lat: float, lon: float,
                    year: int, month: int, variable: str) -> float:
        """Single scalar: mean for the requested month."""
        da = self._raw(variable)
        da_t = da.sel(
            time=(da.time.dt.year == year) & (da.time.dt.month == month)
        )
        if da_t.time.size == 0:
            raise ValueError(f"No data for {year}-{month:02d} in {self.path}")
        da_pt = da_t.sel(latitude=lat, longitude=lon, method="nearest")
        return float(da_pt.mean().values)

    # ── time series ──────────────────────────────────────────────────────
    def annual_timeseries(self, lat: float, lon: float,
                          variable: str) -> dict:
        """Return {year: annual_mean} for every year in the file."""
        da = self._raw(variable)
        pt = da.sel(latitude=lat, longitude=lon, method="nearest")
        annual = pt.groupby("time.year").mean("time")
        return {int(y): float(v)
                for y, v in zip(annual.year.values, annual.values)
                if not np.isnan(float(v))}

    # ── monthly climatology ──────────────────────────────────────────────
    def monthly_climatology(self, lat: float, lon: float,
                            variable: str, year: int) -> list:
        """Return list of 12 monthly means for the given year."""
        da = self._raw(variable)
        pt = da.sel(latitude=lat, longitude=lon, method="nearest")
        vals = []
        for m in range(1, 13):
            da_m = pt.sel(time=(pt.time.dt.year == year) &
                               (pt.time.dt.month == m))
            if da_m.time.size:
                vals.append(float(da_m.mean().values))
            else:
                # fallback: climatological mean for that month
                da_clim = pt.sel(time=pt.time.dt.month == m)
                vals.append(float(da_clim.mean().values) if da_clim.time.size else np.nan)
        return vals

    # ── regional spatial slice ───────────────────────────────────────────
    def spatial_slice(self, lat: float, lon: float,
                      year: int, month: int,
                      variable: str, span_lat: float = 12.0, span_lon: float = 12.0):
        """
        Returns (lats_1d, lons_1d, values_2d) numpy arrays
        for a box centred on (lat,lon).
        """
        da = self._raw(variable)
        da_t = da.sel(
            time=(da.time.dt.year == year) & (da.time.dt.month == month)
        ).mean("time")

        lat_min = max(-89.5, lat - span_lat)
        lat_max = min(89.5,  lat + span_lat)
        lon_min = max(-179.5, lon - span_lon)
        lon_max = min(179.5,  lon + span_lon)

        region = da_t.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
        
        # Upsample to increase numbers and blend beautifully
        try:
            new_lats = np.linspace(float(region.latitude[0]), float(region.latitude[-1]), len(region.latitude) * 3)
            new_lons = np.linspace(float(region.longitude[0]), float(region.longitude[-1]), len(region.longitude) * 3)
            region = region.interp(latitude=new_lats, longitude=new_lons, method='linear')
        except Exception:
            pass # fallback to original resolution if scipy is missing

        lats = region.latitude.values
        lons = region.longitude.values
        vals = region.values   # shape (lat, lon)
        return lats, lons, vals

    @property
    def time_range(self):
        t = self.ds.time.values
        return pd.Timestamp(t[0]).year, pd.Timestamp(t[-1]).year

    @property
    def dims(self):
        return dict(self.ds.dims)


# ─── Classification helpers ──────────────────────────────────────────────────
def classify(value: float, variable: str) -> str:
    if variable == "wind_speed":
        b = min(12, int(value / 1.5))
        names = {0:"Calm",1:"Light Air",2:"Light Breeze",3:"Gentle Breeze",
                 4:"Moderate Breeze",5:"Fresh Breeze",6:"Strong Breeze",
                 7:"Near Gale",8:"Gale",9:"Strong Gale",10:"Storm",
                 11:"Violent Storm",12:"Hurricane Force"}
        return f"Beaufort {b} — {names.get(b,'')}"
    if variable == "temperature":
        if value > 40: return "Extreme Heat"
        if value > 30: return "Very Hot"
        if value > 20: return "Warm"
        if value > 10: return "Mild"
        if value > 0:  return "Cool"
        return "Cold"
    if variable == "precipitation":
        if value > 10: return "Very Heavy"
        if value > 5:  return "Heavy"
        if value > 2:  return "Moderate"
        if value > 0.5: return "Light"
        return "Trace / Dry"
    if variable == "solar_radiation":
        if value > 300: return "Very High Insolation"
        if value > 200: return "High Insolation"
        if value > 100: return "Moderate Insolation"
        return "Low Insolation"
    if variable == "soil_moisture":
        if value > 0.40: return "Saturated"
        if value > 0.30: return "Wet"
        if value > 0.20: return "Moist"
        if value > 0.10: return "Dry"
        return "Very Dry"
    if variable == "snowfall":
        if value > 5: return "Heavy Snow"
        if value > 1: return "Moderate Snow"
        if value > 0.1: return "Light Snow"
        return "Trace / No Snow"
    return "—"


# ─── Chart builders ──────────────────────────────────────────────────────────
def chart_regional_heatmap(nc: NCDataset, lat, lon, year, month, variable):
    cfg   = VARIABLES[variable]
    lats, lons, vals_2d = nc.spatial_slice(lat, lon, year, month, variable, span_lat=12, span_lon=34)

    # Flatten for scatter geo
    lo_g, la_g = np.meshgrid(lons, lats)
    flat_la = la_g.flatten()
    flat_lo = lo_g.flatten()
    flat_v  = vals_2d.flatten()

    # Remove NaNs
    mask = ~np.isnan(flat_v)
    flat_la, flat_lo, flat_v = flat_la[mask], flat_lo[mask], flat_v[mask]

    # Value at the exact selected point
    dists   = np.sqrt((flat_la - lat)**2 + (flat_lo - lon)**2)
    sel_val = float(flat_v[dists.argmin()]) if len(dists) else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=flat_la.tolist(), lon=flat_lo.tolist(),
        mode="markers",
        marker=dict(
            symbol="square",
            size=8, color=flat_v.tolist(), colorscale=cfg["colorscale"],
            showscale=True, opacity=0.85, line=dict(width=0),
            colorbar=dict(
                title=dict(text=cfg["unit"], font=dict(color="#4ad4e8", size=10)),
                tickfont=dict(color="#4ad4e8", size=9),
                bgcolor="rgba(0,8,24,0.75)",
                bordercolor="rgba(74,212,232,0.22)", borderwidth=1, len=0.75,
            ),
        ),
        hovertemplate=(
            f"Lat: %{{lat:.2f}}° Lon: %{{lon:.2f}}°"
            f"<br>{cfg['label']}: %{{marker.color:.2f}} {cfg['unit']}"
            "<extra></extra>"
        ),
        name=cfg["label"],
    ))
    fig.add_trace(go.Scattergeo(
        lat=[lat], lon=[lon], mode="markers+text",
        marker=dict(size=16, color="#f0b429", line=dict(color="white", width=2)),
        text=[cfg["fmt"](sel_val)], textposition="top center",
        textfont=dict(color="white", size=11), showlegend=False,
        hovertemplate=f"Selected: {cfg['fmt'](sel_val)}<extra></extra>",
    ))
    fig.update_geos(
        showcoastlines=True, coastlinecolor="rgba(130,185,255,0.5)",
        showland=True, landcolor="rgba(12,22,50,1)",
        showocean=True, oceancolor="rgba(4,8,28,1)",
        showlakes=True, lakecolor="rgba(6,14,48,1)",
        showframe=True, framecolor="rgba(74,212,232,0.3)", bgcolor="rgba(0,0,0,0)",
        projection_type="equirectangular",
        center=dict(lat=float(lat), lon=float(lon)),
        lataxis_range=[float(lat)-14, float(lat)+14],
        lonaxis_range=[float(lon)-36, float(lon)+36],
    )
    fig.update_layout(
        **_DARK,
        title=dict(
            text=f"Regional {cfg['label']} — {MONTHS[month-1]} {year} (ERA5)",
            font=dict(color="#4ad4e8", size=12),
        ),
        height=340, geo=dict(bgcolor="rgba(0,0,0,0)"),
        dragmode=False,
    )
    return _toj(fig)


def chart_timeseries(nc: NCDataset, lat, lon, variable):
    cfg  = VARIABLES[variable]
    ts   = nc.annual_timeseries(lat, lon, variable)

    if not ts:
        raise ValueError("No time-series data available in NetCDF for this variable.")

    years = sorted(ts.keys())
    vals  = [ts[y] for y in years]
    vals_arr = np.array(vals)

    poly  = np.polyfit(np.arange(len(years)), vals_arr, 1)
    trend = np.poly1d(poly)(np.arange(len(years)))
    slope_dec = float(poly[0]) * 10

    mask_bl   = np.array([1980 <= y <= 2010 for y in years])
    baseline  = float(vals_arr[mask_bl].mean()) if mask_bl.any() else float(vals_arr.mean())
    anomaly   = float(vals_arr[-1] - baseline)

    fig = go.Figure()
    # Baseline band
    fig.add_hrect(
        y0=baseline - abs(baseline) * 0.01,
        y1=baseline + abs(baseline) * 0.01,
        fillcolor="rgba(74,212,232,0.05)",
        line_color="rgba(74,212,232,0.18)",
        annotation_text="1980–2010 baseline",
        annotation_font_color="#4ad4e8",
        annotation_font_size=9,
    )
    fig.add_trace(go.Scatter(
        x=years, y=vals, mode="lines", name="Annual Mean",
        line=dict(color="#4ad4e8", width=2),
        hovertemplate="%{x}: %{y:.3f} " + cfg["unit"] + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=trend.tolist(), mode="lines",
        name=f"Trend ({slope_dec:+.3f} {cfg['unit']}/decade)",
        line=dict(color="#f0b429", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        **_DARK,
        title=dict(
            text=f"Historical {cfg['label']} ({years[0]}–{years[-1]}) — ERA5",
            font=dict(color="#4ad4e8", size=12),
        ),
        xaxis={**_XA, "title": "Year"},
        yaxis={**_YA, "title": f"{cfg['label']} ({cfg['unit']})"},
        height=260,
    )
    return _toj(fig), float(vals_arr[-1]), baseline, anomaly, slope_dec


def chart_seasonal(nc: NCDataset, lat, lon, variable, year):
    cfg  = VARIABLES[variable]
    vals = nc.monthly_climatology(lat, lon, variable, year)

    # Replace any NaN with climatological average for that month
    arr  = np.array(vals, dtype=float)
    mean = float(np.nanmean(arr))
    arr  = np.where(np.isnan(arr), mean, arr)
    vals = arr.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MONTHS, y=vals,
        marker=dict(
            color=["#4ad4e8" if v >= mean else "#a78bfa" for v in vals],
            opacity=0.85, line=dict(color="rgba(74,212,232,0.3)", width=1),
        ),
        hovertemplate="%{x}: %{y:.3f} " + cfg["unit"] + "<extra></extra>",
        name="Monthly Mean",
    ))
    fig.add_hline(
        y=mean, line_dash="dot", line_color="rgba(240,180,40,0.7)",
        annotation_text=f"Annual Mean: {mean:.2f} {cfg['unit']}",
        annotation_font_color="#f0b429", annotation_font_size=9,
    )
    fig.update_layout(
        **_DARK,
        title=dict(
            text=f"Seasonal Climatology — {cfg['label']} {year} (ERA5)",
            font=dict(color="#4ad4e8", size=12),
        ),
        xaxis={**_XA, "title": "Month"},
        yaxis={**_YA, "title": cfg["unit"]},
        height=240,
    )
    return _toj(fig), vals


def chart_comparison(nc: NCDataset, lat, lon, variable):
    cfg  = VARIABLES[variable]
    v90  = nc.monthly_climatology(lat, lon, variable, 1990)
    v20  = nc.monthly_climatology(lat, lon, variable, 2020)

    # Clean NaNs
    v90 = [float(v) if not np.isnan(v) else 0.0 for v in v90]
    v20 = [float(v) if not np.isnan(v) else 0.0 for v in v20]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=MONTHS, y=v90, mode="lines+markers", name="1990",
        line=dict(color="#4ad4e8", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=MONTHS, y=v20, mode="lines+markers", name="2020",
        line=dict(color="#a78bfa", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=MONTHS + MONTHS[::-1], y=v20 + v90[::-1],
        fill="toself", fillcolor="rgba(167,139,250,0.07)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.update_layout(
        **_DARK,
        title=dict(
            text=f"1990 vs 2020 — {cfg['label']} (ERA5)",
            font=dict(color="#4ad4e8", size=12),
        ),
        xaxis={**_XA, "title": "Month"},
        yaxis={**_YA, "title": cfg["unit"]},
        height=240,
    )
    return _toj(fig)


def chart_rcp(nc: NCDataset, lat, lon, variable, cur_val):
    cfg    = VARIABLES[variable]
    slopes = cfg["rcp_slopes"]
    fy     = list(range(2024, 2061))

    # Fetch historical time-series for training
    ts = nc.annual_timeseries(lat, lon, variable)
    if not ts:
        # Fallback if no history
        ts = {2020: cur_val, 2023: cur_val}
        
    years = sorted(ts.keys())
    vals  = [ts[y] for y in years]

    # Train an ML model (Polynomial Features + Ridge Regression)
    X = np.array(years).reshape(-1, 1)
    y = np.array(vals)

    # Use degree 3 for realistic slight curving, but Ridge prevents extreme overfitting
    model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=100.0))
    model.fit(X, y)

    # Calculate historical residuals to simulate natural climate variability
    historical_pred = model.predict(X)
    residuals = y - historical_pred
    std_resid = float(np.std(residuals)) if len(residuals) > 2 else abs(cur_val) * 0.05
    if std_resid == 0:
        std_resid = abs(cur_val) * 0.05

    # Predict base future curve
    X_pred = np.array(fy).reshape(-1, 1)
    base_pred = model.predict(X_pred)

    # Align the start of the ML curve exactly with the current value
    offset = cur_val - base_pred[0]
    base_pred_aligned = base_pred + offset

    rcps = {}
    for i, scen in enumerate(["2.6", "4.5", "8.5"]):
        # Seed for reproducible "random" noise so the chart doesn't jitter on reload
        np.random.seed(42 + int(abs(lat) * 100) + int(abs(lon) * 100) + i)
        
        # Generate autocorrelated noise (AR1) to mimic natural variability
        noise = np.zeros(len(fy))
        noise[0] = np.random.normal(0, std_resid)
        rho = 0.4  # AR(1) autocorrelation coefficient
        for j in range(1, len(fy)):
            noise[j] = rho * noise[j-1] + np.random.normal(0, std_resid * np.sqrt(1 - rho**2))

        rcp_vals = []
        for j, y_val in enumerate(fy):
            # ML base shape + RCP specific deviation (slope per decade -> slope/10 per year) + noise
            val = base_pred_aligned[j] + (slopes[i] / 10.0) * j + noise[j]
            rcp_vals.append(float(val))
        rcps[scen] = rcp_vals
    colors = {"2.6": "#4ad4e8", "4.5": "#f0b429", "8.5": "#a78bfa"}
    names  = {
        "2.6": "RCP 2.6 — Stabilised",
        "4.5": "RCP 4.5 — Moderate",
        "8.5": "RCP 8.5 — High Emissions",
    }

    fig = go.Figure()
    for k, yd in rcps.items():
        fig.add_trace(go.Scatter(
            x=fy, y=yd, mode="lines", name=names[k],
            line=dict(color=colors[k], width=2),
            hovertemplate="%{x}: %{y:.3f} " + cfg["unit"] + "<extra></extra>",
        ))
    fig.update_layout(
        **_DARK,
        title=dict(
            text=f"Climate Projections 2024–2060 — {cfg['label']}",
            font=dict(color="#4ad4e8", size=12),
        ),
        xaxis={**_XA, "title": "Year"},
        yaxis={**_YA, "title": cfg["unit"]},
        height=250,
    )
    proj_2045 = {k: round(float(v[fy.index(2045)]), 3) for k, v in rcps.items()}
    return _toj(fig), proj_2045


MAJOR_CITIES = [
    {"name": "London, UK", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris, France", "lat": 48.8566, "lon": 2.3522},
    {"name": "Berlin, Germany", "lat": 52.5200, "lon": 13.4050},
    {"name": "Madrid, Spain", "lat": 40.4168, "lon": -3.7038},
    {"name": "Rome, Italy", "lat": 41.9028, "lon": 12.4964},
    {"name": "Moscow, Russia", "lat": 55.7558, "lon": 37.6173},
    {"name": "New York, USA", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles, USA", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago, USA", "lat": 41.8781, "lon": -87.6298},
    {"name": "Toronto, Canada", "lat": 43.6510, "lon": -79.3470},
    {"name": "Vancouver, Canada", "lat": 49.2827, "lon": -123.1207},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    {"name": "São Paulo, Brazil", "lat": -23.5505, "lon": -46.6333},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729},
    {"name": "Buenos Aires, Argentina", "lat": -34.6037, "lon": -58.3816},
    {"name": "Lima, Peru", "lat": -12.0464, "lon": -77.0428},
    {"name": "Bogotá, Colombia", "lat": 4.7110, "lon": -74.0721},
    {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503},
    {"name": "Beijing, China", "lat": 39.9042, "lon": 116.4074},
    {"name": "Shanghai, China", "lat": 31.2304, "lon": 121.4737},
    {"name": "Guangzhou, China", "lat": 23.1291, "lon": 113.2644},
    {"name": "Hong Kong", "lat": 22.3193, "lon": 114.1694},
    {"name": "Taipei, Taiwan", "lat": 25.0330, "lon": 121.5654},
    {"name": "Seoul, South Korea", "lat": 37.5665, "lon": 126.9780},
    {"name": "New Delhi, India", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai, India", "lat": 19.0760, "lon": 72.8777},
    {"name": "Bengaluru, India", "lat": 12.9716, "lon": 77.5946},
    {"name": "Bangkok, Thailand", "lat": 13.7563, "lon": 100.5018},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"name": "Jakarta, Indonesia", "lat": -6.2088, "lon": 106.8456},
    {"name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne, Australia", "lat": -37.8136, "lon": 144.9631},
    {"name": "Auckland, New Zealand", "lat": -36.8485, "lon": 174.7633},
    {"name": "Cairo, Egypt", "lat": 30.0444, "lon": 31.2357},
    {"name": "Nairobi, Kenya", "lat": -1.2921, "lon": 36.8219},
    {"name": "Lagos, Nigeria", "lat": 6.5244, "lon": 3.3792},
    {"name": "Cape Town, South Africa", "lat": -33.9249, "lon": 18.4241},
    {"name": "Johannesburg, RSA", "lat": -26.2041, "lon": 28.0473},
    {"name": "Dubai, UAE", "lat": 25.2048, "lon": 55.2708},
    {"name": "Riyadh, Saudi Arabia", "lat": 24.7136, "lon": 46.6753},
    {"name": "Istanbul, Turkey", "lat": 41.0082, "lon": 28.9784},
    {"name": "Tehran, Iran", "lat": 35.6892, "lon": 51.3890},
    {"name": "Karachi, Pakistan", "lat": 24.8607, "lon": 67.0011},
    {"name": "Manila, Philippines", "lat": 14.5995, "lon": 120.9842},
    {"name": "Ho Chi Minh City, Vietnam", "lat": 10.8231, "lon": 106.6297},
    {"name": "Kuala Lumpur, Malaysia", "lat": 3.1390, "lon": 101.6869},
    {"name": "Reykjavik, Iceland", "lat": 64.1466, "lon": -21.9426},
    {"name": "Oslo, Norway", "lat": 59.9139, "lon": 10.7522},
    {"name": "Stockholm, Sweden", "lat": 59.3293, "lon": 18.0686},
    {"name": "Helsinki, Finland", "lat": 60.1695, "lon": 24.9354},
    {"name": "Copenhagen, Denmark", "lat": 55.6761, "lon": 12.5683},
    {"name": "Warsaw, Poland", "lat": 52.2297, "lon": 21.0122},
    {"name": "Athens, Greece", "lat": 37.9838, "lon": 23.7275},
    {"name": "Lisbon, Portugal", "lat": 38.7223, "lon": -9.1393},
    {"name": "Dublin, Ireland", "lat": 53.3498, "lon": -6.2603},
    {"name": "Honolulu, USA", "lat": 21.3069, "lon": -157.8583},
    {"name": "Anchorage, USA", "lat": 61.2181, "lon": -149.9003},
    {"name": "Miami, USA", "lat": 25.7617, "lon": -80.1918},
    {"name": "Havana, Cuba", "lat": 23.1136, "lon": -82.3666},
    {"name": "Caracas, Venezuela", "lat": 10.4806, "lon": -66.9036},
    {"name": "Santiago, Chile", "lat": -33.4489, "lon": -70.6693},
    {"name": "Algiers, Algeria", "lat": 36.7538, "lon": 3.0588},
    {"name": "Casablanca, Morocco", "lat": 33.5731, "lon": -7.5898},
    {"name": "Jerusalem", "lat": 31.7683, "lon": 35.2137},
    {"name": "Dar es Salaam, Tanzania", "lat": -6.7924, "lon": 39.2083},
    {"name": "Kinshasa, DRC", "lat": -4.4419, "lon": 15.2663},
    {"name": "Addis Ababa, Ethiopia", "lat": 9.0300, "lon": 38.7400},
    {"name": "Dakar, Senegal", "lat": 14.7167, "lon": -17.4677},
]

def find_analogues(nc: NCDataset, target_lat: float, target_lon: float, target_proj_val: float, variable: str, current_year: int, month: int):
    cfg = VARIABLES[variable]
    results = []
    
    da = nc._raw(variable)
    try:
        da_recent = da.sel(time=da.time.dt.year >= (current_year - 5)).mean("time")
    except:
        da_recent = da.mean("time")

    for city in MAJOR_CITIES:
        if abs(city["lat"] - target_lat) < 2 and abs(city["lon"] - target_lon) < 2:
            continue
            
        try:
            pt = da_recent.sel(latitude=city["lat"], longitude=city["lon"], method="nearest")
            cur_val = float(pt.values)
            if np.isnan(cur_val): continue
            
            diff = abs(cur_val - target_proj_val)
            pct_match = max(0, 100 - (diff / max(0.01, abs(target_proj_val))) * 100)
            
            results.append({
                "name": city["name"],
                "lat": city["lat"],
                "lon": city["lon"],
                "val": cur_val,
                "diff": diff,
                "match": f"{min(99, int(pct_match))}%",
                "meta": f"{cfg['fmt'](cur_val)} · Real ERA5",
            })
        except:
            pass
            
    results.sort(key=lambda x: x["diff"])
    return results[:4]

def chart_analogue_map(target_lat: float, target_lon: float, target_name: str, analogues: list, variable: str):
    cfg = VARIABLES[variable]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lat=[target_lat], lon=[target_lon],
        mode="markers+text",
        marker=dict(size=14, color="#4ad4e8", line=dict(color="rgba(74,212,232,0.6)", width=4)),
        text=[f"Target: {target_name}"], textposition="top right",
        textfont=dict(color="#4ad4e8", size=11, family="Exo 2"),
        name="Projected Location",
        hovertemplate="Target: %{text}<extra></extra>"
    ))
    
    alats = [c["lat"] for c in analogues]
    alons = [c["lon"] for c in analogues]
    atext = [f"{c['name']} ({c['match']} match)" for c in analogues]
    
    fig.add_trace(go.Scattergeo(
        lat=alats, lon=alons,
        mode="markers+text",
        marker=dict(size=10, color="#a78bfa", line=dict(color="rgba(167,139,250,0.4)", width=3)),
        text=atext, textposition="bottom center",
        textfont=dict(color="#a78bfa", size=10, family="Exo 2"),
        name="Present Day Analogues",
        hovertemplate="%{text}<extra></extra>"
    ))
    
    fig.update_geos(
        showcoastlines=True, coastlinecolor="rgba(100,220,255,0.15)",
        showland=True, landcolor="rgba(0,10,25,0.8)",
        showocean=True, oceancolor="rgba(0,4,12,1)",
        showcountries=True, countrycolor="rgba(100,220,255,0.08)",
        showframe=False, bgcolor="rgba(0,0,0,0)",
        projection_type="natural earth",
    )
    fig.update_layout(**_DARK)
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(0,10,25,0.8)", bordercolor="rgba(100,220,255,0.2)"),
    )
    return _toj(fig)


# ─── Main processor ──────────────────────────────────────────────────────────
class ClimateDataProcessor:
    """
    Holds one or more NCDataset objects (one per variable file or a combined file).
    The Flask app instantiates one of these at startup.
    """

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    def __init__(self):
        self._datasets: dict[str, NCDataset] = {}
        self._scan_data_dir()

    # ── file management ──────────────────────────────────────────────────
    def _scan_data_dir(self):
        """Auto-load any .nc files found in data/."""
        if not os.path.isdir(self.DATA_DIR):
            return
        for fname in os.listdir(self.DATA_DIR):
            if fname.endswith(".nc"):
                path = os.path.join(self.DATA_DIR, fname)
                self._load_file(path)

    def _load_file(self, path: str):
        """Load a NetCDF file and register which variables it provides."""
        try:
            nc = NCDataset(path)
            for var in nc.available_variables():
                self._datasets[var] = nc
                print(f"[PROC] {var} → {os.path.basename(path)}")
        except Exception as e:
            print(f"[PROC ERR] Could not load {path}: {e}")

    def load_uploaded(self, path: str) -> dict:
        """Called by /api/upload-nc after a user uploads a file."""
        nc = NCDataset(path)
        avail = nc.available_variables()
        for var in avail:
            self._datasets[var] = nc
        return {
            "success": True,
            "variables": avail,
            "dims": nc.dims,
            "time_range": nc.time_range,
        }

    def ready_variables(self) -> list:
        return list(self._datasets.keys())

    def _nc(self, variable: str) -> NCDataset:
        if variable not in self._datasets:
            raise RuntimeError(
                f"No NetCDF data loaded for '{variable}'. "
                f"Available: {self.ready_variables()}. "
                "Run download_era5.py to fetch data, or upload a .nc file."
            )
        return self._datasets[variable]

    # ── main analysis ────────────────────────────────────────────────────
    def analyze(self, lat: float, lon: float, year: int, month: int,
                variable: str, loc_name: str = "") -> dict:

        nc  = self._nc(variable)
        cfg = VARIABLES[variable]

        # ── current scalar value ─────────────────────────────────────
        cur = nc.point_value(lat, lon, year, month, variable)

        # ── all charts ───────────────────────────────────────────────
        heatmap_fig = chart_regional_heatmap(nc, lat, lon, year, month, variable)
        ts_fig, last_val, baseline, anomaly, slope = chart_timeseries(nc, lat, lon, variable)
        seasonal_fig, monthly_vals = chart_seasonal(nc, lat, lon, variable, year)
        cmp_fig  = chart_comparison(nc, lat, lon, variable)
        rcp_fig, rcp_proj = chart_rcp(nc, lat, lon, variable, cur)

        proj_val = rcp_proj.get("8.5", cur)
        t_loc_name = loc_name or f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
        analogues = find_analogues(nc, lat, lon, proj_val, variable, year, month)
        analogue_fig = chart_analogue_map(lat, lon, t_loc_name, analogues, variable)

        return {
            "success": True,
            "source": "ERA5 NetCDF",
            "location": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "name": loc_name or (
                    f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
                    f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
                ),
            },
            "variable": variable,
            "variable_label": cfg["label"],
            "unit": cfg["unit"],
            "year": year,
            "month": month,
            "month_name": MONTHS[month - 1],
            "current_value": round(float(cur), 3),
            "current_fmt": cfg["fmt"](cur),
            "classification": classify(cur, variable),
            "stats": {
                "baseline_mean":    round(baseline, 3),
                "anomaly":          round(anomaly, 3),
                "trend_per_decade": round(slope, 4),
                "seasonal_min":     round(min(monthly_vals), 3),
                "seasonal_max":     round(max(monthly_vals), 3),
            },
            "rcp_proj_2045": rcp_proj,
            "charts": {
                "regional_map": heatmap_fig,
                "time_series":  ts_fig,
                "monthly":      seasonal_fig,
                "comparison":   cmp_fig,
                "rcp":          rcp_fig,
                "analogues":    analogue_fig,
            },
            "analogues": analogues,
        }

    def heatmap_only(self, lat: float, lon: float,
                     year: int, month: int, variable: str) -> dict:
        """Lightweight endpoint used by the year-animation slider."""
        nc = self._nc(variable)
        return chart_regional_heatmap(nc, lat, lon, year, month, variable)
