from __future__ import annotations

import html
from datetime import datetime


def make_report(quality, index_name: str, start, end, model_results=None) -> str:
    rows = "".join(f"<li>{html.escape(warning)}</li>" for warning in quality.warnings) or "<li>No quality warnings recorded.</li>"
    result = ""
    if model_results:
        result = "<h2>Model evaluation</h2><ul>" + "".join(
            f"<li>{html.escape(name)}: RMSE {values['metrics']['rmse']:.6f}, directional accuracy {values['metrics']['directional_accuracy']:.1%}</li>"
            for name, values in model_results.items()
        ) + "</ul>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>MarketPulse India Report</title>
    <style>body{{font-family:Arial;max-width:900px;margin:40px auto;color:#17324d}} h1{{color:#087f8c}}</style></head><body>
    <h1>MarketPulse India Analysis Report</h1><p>Generated {datetime.now():%Y-%m-%d %H:%M}</p>
    <p><b>Index:</b> {html.escape(index_name)} &nbsp; <b>Period:</b> {start} to {end}</p>
    <p><b>Source:</b> {html.escape(quality.source)} &nbsp; <b>Rows:</b> {quality.rows_cleaned}</p>
    <h2>Data quality</h2><ul>{rows}</ul>{result}
    <h2>Limitations</h2><p>This educational prototype uses historical observations only. Predictions are not financial advice and do not guarantee future results.</p>
    </body></html>"""

