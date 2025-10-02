# noventis/utils/templates.py

from IPython.display import HTML

# Semua CSS dari NoventisAutoEDA disalin ke sini untuk tampilan yang konsisten.
REPORT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Exo+2:wght@600;800&display=swap');

:root { 
    --bg-dark-1: #0D1117; 
    --bg-dark-2: #161B22; 
    --bg-dark-3: #010409; 
    --border-color: #30363D; 
    --text-light: #C9D1D9; 
    --text-muted: #8B949E; 
    --primary-blue: #58A6FF; 
    --primary-orange: #F78166; 
    --font-main: 'Roboto', sans-serif; 
    --font-header: 'Exo 2', sans-serif;
}

body { 
    font-family: var(--font-main); 
    background-color: var(--bg-dark-1); 
    color: var(--text-light); 
    margin: 0; 
    padding: 0; 
}

.container { 
    width: 100%; 
    max-width: 1400px; 
    margin: auto; 
    background-color: var(--bg-dark-1); 
}

header { 
    padding: 1.5rem 2.5rem; 
    border-bottom: 1px solid var(--border-color); 
    background-color: var(--bg-dark-2); 
    border-radius: 10px 10px 0 0;
}

header h1 { 
    font-family: var(--font-header); 
    font-size: 2.5rem; 
    margin: 0; 
    color: var(--primary-blue); 
}

header p { 
    margin: 0.25rem 0 0; 
    color: var(--text-muted); 
    font-size: 1rem; 
}

main { 
    padding: 2.5rem; 
}

h2, h3, h4 { 
    font-family: var(--font-header); 
}

h2 { 
    font-size: 2rem; 
    color: var(--primary-orange); 
    border-bottom: 1px solid var(--border-color); 
    padding-bottom: 0.5rem; 
    margin-top: 0; 
}

h3 { 
    color: var(--primary-blue); 
    font-size: 1.5rem; 
    margin-top: 2rem; 
}

.grid-container { 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
    gap: 1.5rem; 
    margin-bottom: 2rem; 
}

.grid-item { 
    background-color: var(--bg-dark-2); 
    padding: 1.5rem; 
    border-radius: 8px; 
    border: 1px solid var(--border-color); 
}

.grid-item h4 {
    margin-top: 0;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
}

.grid-item p {
    font-size: 1.2rem;
    font-weight: bold;
    margin: 0.5rem 0;
}

.table-scroll-wrapper { 
    margin-top: 1rem; 
    overflow: auto; 
    max-height: 400px;
}

.styled-table, .styled-table-small { 
    width: 100%; 
    color: var(--text-muted); 
    background-color: var(--bg-dark-2); 
    border-collapse: collapse; 
    border-radius: 8px; 
    overflow: hidden; 
    font-size: 0.9rem; 
}

.styled-table th, .styled-table td, 
.styled-table-small th, .styled-table-small td { 
    border-bottom: 1px solid var(--border-color); 
    padding: 0.8rem 1rem; 
    text-align: left; 
    white-space: nowrap; 
}

.styled-table thead th, .styled-table-small thead th { 
    background-color: var(--bg-dark-3); 
    color: var(--text-light);
}

.plot-container {
    background-color: var(--bg-dark-2); 
    padding: 1rem; 
    margin-top: 1rem; 
    border-radius: 8px; 
    border: 1px solid var(--border-color); 
    text-align: center;
}

.plot-container img {
    max-width: 90%; 
    height: auto; 
    border-radius: 5px;
}
"""

def generate_standalone_report(title: str, subtitle: str, content_html: str) -> HTML:
    """
    Membungkus konten laporan (KPI, tabel, plot) dengan kerangka HTML dan CSS standar.
    """
    full_html = f\"\"\"
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>{REPORT_CSS}</style>
    </head>
    <body>
        <div class="container" style="padding: 1.5rem;">
            <header>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </header>
            <main>
                {content_html}
            </main>
        </div>
    </body>
    </html>
    \"\"\"
    return HTML(full_html)