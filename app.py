
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash

# --- Config ---
UPLOAD = "static/plots"
SAMPLE_CSV = "sample_data/weather.csv"   

os.makedirs(UPLOAD, exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret123"  # ganti dengan secret yang aman untuk produksi

# ------------------------
# Utility
# ------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# =============================
# Monte Carlo Functions
# =============================
def monte_parametric(values, nsim=10000):
    diffs = np.diff(values)
    if len(diffs) == 0:
        # fallback jika hanya 1 nilai
        mu, sigma = 0.0, 1.0
    else:
        mu, sigma = diffs.mean(), diffs.std(ddof=1) if len(diffs) > 1 else (diffs.mean(), 0.0)
    sims = np.random.normal(mu, sigma, nsim)
    preds = values[-1] + sims
    return preds

def monte_empirical(values, nsim=10000):
    diffs = np.diff(values)
    if len(diffs) == 0:
        sims = np.zeros(nsim)
    else:
        sims = np.random.choice(diffs, nsim, replace=True)
    preds = values[-1] + sims
    return preds

def save_plot(preds, method):
    plt.figure(figsize=(8,4))
    plt.hist(preds, bins=40)
    plt.title(f"Monte Carlo Prediction — {method}")
    plt.xlabel("Predicted Value")
    plt.ylabel("Frequency")

    filename = f"{method.replace(' ','_')}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(UPLOAD, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return filepath

# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        # ambil pengaturan form
        method = request.form.get("method", "parametric")
        try:
            nsim = int(request.form.get("nsim", 10000))
            if nsim <= 0:
                nsim = 10000
        except Exception:
            nsim = 10000

        use_sample = request.form.get("use_sample", "no")

        # Load data: either sample CSV atau file upload
        df = None
        if use_sample == "yes":
            if not os.path.exists(SAMPLE_CSV):
                flash("Sample CSV tidak ditemukan di server.")
                return redirect(request.url)
            try:
                df = pd.read_csv(SAMPLE_CSV)
            except Exception as e:
                flash(f"Gagal membaca sample CSV: {e}")
                return redirect(request.url)
        else:
            # cek keberadaan input file
            if 'file' not in request.files:
                flash("Form tidak mengirim file. Pastikan form memiliki enctype='multipart/form-data' dan input name='file'.")
                return redirect(request.url)

            file = request.files['file']
            if file.filename == "":
                flash("Nama file kosong. Silakan pilih file CSV.")
                return redirect(request.url)

            if not allowed_file(file.filename):
                flash("Tipe file tidak diizinkan. Gunakan file .csv")
                return redirect(request.url)

            # simpan sementara (opsional) dan baca dengan pandas
            filename = secure_filename(file.filename)
            # baca langsung tanpa menyimpan ke disk
            try:
                df = pd.read_csv(file)
            except Exception as e:
                flash(f"Gagal membaca CSV yang diupload: {e}")
                return redirect(request.url)

        # Validasi kolom
        if "temp" not in df.columns:
            flash("CSV harus memiliki kolom 'temp' (nama case-sensitive).")
            return redirect(request.url)

        # ambil nilai dan cek panjang
        values = df["temp"].dropna().astype(float).values
        if len(values) < 1:
            flash("CSV tidak memiliki nilai suhu yang valid.")
            return redirect(request.url)

        # jalankan simulasi
        if method == "parametric":
            preds = monte_parametric(values, nsim)
            mname = "Parametric Normal"
        else:
            preds = monte_empirical(values, nsim)
            mname = "Empirical Bootstrap"

        # statistik ringkasan
        stats = {
            "mean": float(np.mean(preds)),
            "median": float(np.median(preds)),
            "p10": float(np.percentile(preds,10)),
            "p25": float(np.percentile(preds,25)),
            "p75": float(np.percentile(preds,75)),
            "p90": float(np.percentile(preds,90)),
            "std": float(np.std(preds, ddof=1))
        }

        # simpan plot
        plot = save_plot(preds, mname)

        # render hasil
        return render_template("result.html",
                               method_name=mname,
                               stats=stats,
                               plot_url=url_for('static', filename=f"plots/{os.path.basename(plot)}")
                               )

    # GET request -> tampilkan form
    return render_template("index.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
