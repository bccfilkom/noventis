# noventis/utils/plotting.py

import io
import base64
import matplotlib.pyplot as plt

def plot_to_base64(fig):
    """
    Mengubah objek figure Matplotlib menjadi string Base64 
    untuk disematkan langsung di HTML.
    """
    if fig is None:
        return ""
    
    buf = io.BytesIO()
    # Menyimpan figure ke buffer memori sebagai PNG
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    # Mengenkode gambar menjadi string base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    # Menutup figure untuk melepaskan memori
    plt.close(fig)
    
    return f"data:image/png;base64,{img_str}"

def set_noventis_theme():
    """
    Mengatur tema plot standar (dark mode) untuk semua visualisasi 
    agar konsisten di seluruh library.
    """
    plt.rcParams.update({
        'font.size': 12, 
        'axes.labelsize': 10, 
        'xtick.labelsize': 8, 
        'ytick.labelsize': 8, 
        'axes.titlesize': 12, 
        'figure.titlesize': 14, 
        'legend.fontsize': 10, 
        'figure.facecolor': '#161B22', 
        'axes.facecolor': '#161B22', 
        'text.color': '#C9D1D9', 
        'axes.labelcolor': '#C9D1D9', 
        'xtick.color': '#8B949E', 
        'ytick.color': '#8B949E',
        'grid.color': '#30363D', 
        'patch.edgecolor': '#30363D', 
        'figure.edgecolor': '#161B22',
    })