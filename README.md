# PyClimaExplorer 📊🌍

**PyClimaExplorer** is a professional climate data analysis and visualization tool built for **TECHNEX'26 at IIT BHU**. It leverages real-world **ERA5 reanalysis data** and **Google Gemini AI** to provide deep insights into climatic trends, anomalies, and future projections.

![PyClimaExplorer Logo](static/logo.png) *(Note: Replace with actual logo path if available)*

## 🚀 Features

-   **Interactive Globe 🌍**: Visualize global climate patterns on an interactive 3D globe.
-   **Regional Heatmaps 🔥**: High-resolution analysis of temperature, precipitation, and more using ERA5 data.
-   **Anomaly Analysis 📉**: Compare current conditions against a 30-year baseline (1980–2010).
-   **Future Projections 🔮**: View climate scenarios for 2045 based on RCP 2.6, 4.5, and 8.5 models.
-   **AI Climate Analyst 🤖**: Get natural language insights powered by **Google Gemini**, context-aware of the local data.
-   **Custom Data Upload 📤**: Support for custom NetCDF (`.nc`) file analysis.

## 🛠️ Technical Stack

-   **Backend**: Flask (Python)
-   **Frontend**: HTML5, CSS3 (Modern Glassmorphic UI), Vanilla JS
-   **Visualization**: Plotly.js
-   **Data Processing**: Xarray, NetCDF4, NumPy, Pandas
-   **AI**: Google Generative AI (Gemini 1.5/2.5 Flash)

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PyClimaExplorer.git
cd PyClimaExplorer
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
FLASK_SECRET_KEY=your_secret_key
FLASK_DEBUG=true
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
CDS_API_KEY=your_copernicus_cds_api_key  # Optional: only if using download_era5.py
```

### 4. Fetch Climate Data
You need ERA5 data in the `data/` folder. You can use the provided script (requires CDS API Key):
```bash
python download_era5.py
```

### 5. Run the Application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

-   `app.py`: Main Flask application entry point.
-   `data_processor.py`: Core logic for climate data analysis and Plotly generation.
-   `download_era5.py`: Script to download reanalysis data from Copernicus CDS.
-   `templates/`: HTML templates (`index.html`, `results.html`).
-   `static/`: CSS, JS, and image assets.
-   `data/`: Directory for NetCDF climate data.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Built with ❤️ for TECHNEX'26, IIT BHU.
