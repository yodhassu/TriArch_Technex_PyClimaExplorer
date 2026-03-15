"""
PyClimaExplorer — Flask Backend  (TECHNEX'26 · IIT BHU)
"""
import os, json, traceback

# ── Load .env FIRST — before anything else reads os.environ ─────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__,
             template_folder=os.path.join(BASE, "templates"),
             static_folder  =os.path.join(BASE, "static"))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# Gemini model read from .env — defaults to gemini-2.5-flash for speed/cost balance
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ── Lazy-load processor ──────────────────────────────────────────────────────
_proc = None
def get_proc():
    global _proc
    if _proc is None:
        from data_processor import ClimateDataProcessor
        _proc = ClimateDataProcessor()
    return _proc

# ── Gemini client (optional — only needed for /api/ai-analysis) ───────────
def get_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMINI_MODEL)
    except ImportError as e:
        print(f"Error loading Gemini: {e} - Please run 'pip install google-generativeai'")
        return None


# ════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    return render_template("results.html")


# ════════════════════════════════════════════════════════════════════════════
#  API — STATUS
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/status")
def status():
    """Frontend polls this to know which variables are loaded."""
    try:
        proc = get_proc()
        ready = proc.ready_variables()
        from data_processor import VARIABLES
        return jsonify({
            "ready": ready,
            "all_variables": [
                {"id": k, "label": VARIABLES[k]["label"], "unit": VARIABLES[k]["unit"]}
                for k in VARIABLES
            ],
            "loaded_variables": [
                {"id": k, "label": VARIABLES[k]["label"], "unit": VARIABLES[k]["unit"]}
                for k in ready
            ],
            "has_data": len(ready) > 0,
        })
    except Exception as e:
        return jsonify({"error": str(e), "has_data": False}), 500


# ════════════════════════════════════════════════════════════════════════════
#  API — UPLOAD NetCDF
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/upload-nc", methods=["POST"])
def upload_nc():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file sent"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".nc"):
        return jsonify({"success": False, "error": "Only .nc files accepted"}), 400

    upload_dir = os.path.join(BASE, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, f.filename)
    f.save(save_path)

    try:
        result = get_proc().load_uploaded(save_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════
#  API — FULL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        b = request.get_json(force=True) or {}
        lat      = float(b.get("lat")      or 25.3)
        lon      = float(b.get("lon")      or 83.0)
        year     = int  (b.get("year")     or 2020)
        month    = int  (b.get("month")    or 6)
        variable = str  (b.get("variable") or "temperature")
        loc_name = str  (b.get("location_name") or "")

        result = get_proc().analyze(lat, lon, year, month, variable, loc_name)
        return jsonify(result)

    except RuntimeError as e:
        # Missing data file — user-facing message
        return jsonify({"success": False,
                         "error": str(e),
                         "error_type": "no_data"}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════
#  API — HEATMAP ONLY  (year-slider animation)
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    """
    Lightweight endpoint — returns only the regional heatmap Plotly JSON.
    Called every time the user drags the year slider.
    """
    try:
        b = request.get_json(force=True) or {}
        lat      = float(b.get("lat")      or 25.3)
        lon      = float(b.get("lon")      or 83.0)
        year     = int  (b.get("year")     or 2020)
        month    = int  (b.get("month")    or 6)
        variable = str  (b.get("variable") or "temperature")

        fig_json = get_proc().heatmap_only(lat, lon, year, month, variable)
        return jsonify({"success": True, "chart": fig_json})

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e), "error_type": "no_data"}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════
#  API — AI ANALYSIS  (Gemini streaming)
# ════════════════════════════════════════════════════════════════════════════
@app.route("/api/ai-analysis", methods=["POST"])
def ai_analysis():
    """
    Accepts climate data context + optional user question.
    Streams back a Gemini response as SSE (text/event-stream).
    """
    model = get_gemini()
    if model is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return jsonify({
                "success": False,
                "error": "GEMINI_API_KEY not set. "
                         "Set it in your environment (.env) and restart the server.",
            }), 503
        else:
            return jsonify({
                "success": False,
                "error": "The 'google.generativeai' Python package is not installed. "
                         "Please run 'pip install google-generativeai' in your terminal and restart the server.",
            }), 503

    b = request.get_json(force=True) or {}

    # Build a rich system prompt from the climate data passed from the frontend
    loc      = b.get("location", {})
    variable = b.get("variable", "temperature")
    unit     = b.get("unit", "")
    cur_val  = b.get("current_value", "")
    cur_fmt  = b.get("current_fmt", "")
    classif  = b.get("classification", "")
    stats    = b.get("stats", {})
    rcp      = b.get("rcp_proj_2045", {})
    year     = b.get("year", "")
    month    = b.get("month_name", "")
    question = b.get("question", "").strip()

    system_prompt = f"""You are an expert climate scientist and data analyst for PyClimaExplorer, 
a professional climate visualisation tool built for TECHNEX'26 at IIT BHU.

You are analysing real ERA5 reanalysis data for the following query:

LOCATION     : {loc.get('name', 'Unknown')} ({loc.get('lat', '')}°, {loc.get('lon', '')}°)
VARIABLE     : {variable} — {b.get('variable_label', '')}
PERIOD       : {month} {year}
CURRENT VALUE: {cur_fmt}
CATEGORY     : {classif}

STATISTICS:
  • Baseline mean (1980–2010) : {stats.get('baseline_mean', '')} {unit}
  • Anomaly vs baseline       : {stats.get('anomaly', '')} {unit}
  • Long-term trend           : {stats.get('trend_per_decade', '')} {unit}/decade
  • Seasonal min/max          : {stats.get('seasonal_min', '')} / {stats.get('seasonal_max', '')} {unit}

CLIMATE PROJECTIONS (2045):
  • RCP 2.6 (aggressive mitigation) : {rcp.get('2.6', '')} {unit}
  • RCP 4.5 (moderate action)       : {rcp.get('4.5', '')} {unit}
  • RCP 8.5 (business-as-usual)     : {rcp.get('8.5', '')} {unit}

Provide concise, scientifically accurate, and engaging analysis.
Reference specific numbers from the data above.
Use plain text — no markdown headers or bullet lists, just flowing paragraphs.
Keep response under 200 words unless asked for more detail.
"""

    user_message = question if question else (
        f"Provide a climate analysis for {loc.get('name', 'this location')} "
        f"based on the data above. Cover the anomaly significance, "
        f"what is driving it, and what the future projections mean."
    )

    def generate():
        try:
            stream = model.generate_content(
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_message}",
                stream=True
            )
            for chunk in stream:
                if getattr(chunk, 'text', None):
                    # SSE format
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n  ✅  PyClimaExplorer  →  http://localhost:{port}")
    print(f"  🤖  Gemini model     : {GEMINI_MODEL}")
    print(f"  🔑  Gemini key       : {'✓ set' if os.environ.get('GEMINI_API_KEY') else '✗ NOT SET — AI analysis disabled'}")
    print(f"  📡  CDS API key      : {'✓ set' if os.environ.get('CDS_API_KEY') else '✗ not set — needed for download_era5.py'}")

    ready = get_proc().ready_variables()
    if ready:
        print(f"  📊  Variables loaded : {', '.join(ready)}")
    else:
        print("  ⚠️   No NetCDF data in data/ — run: python download_era5.py")
    print()
    app.run(debug=debug, port=port, host="0.0.0.0")
