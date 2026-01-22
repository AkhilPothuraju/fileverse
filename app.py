from flask import (
    Flask, render_template, request, redirect,
    flash, send_file, send_from_directory,
    session, jsonify, url_for
)
from dotenv import load_dotenv
load_dotenv()
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import zipfile
import shutil
import sqlite3
import threading
import random
import time
import smtplib
from email.message import EmailMessage
from werkzeug.security import generate_password_hash, check_password_hash
import os
import subprocess
import re
import base64
import io
import fitz
import language_tool_python
from PIL import Image
# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.secret_key = "fileverse_secret_key"

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

UPLOAD_FOLDER = "uploads"
UPLOAD_DIR = "static/blur"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# GLOBAL BLUR STORAGE
# =========================
pages = []
edited_pages = {}
uploaded_file_type = None 

# =========================
# DATABASE
# =========================
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            otp TEXT,
            otp_expiry INTEGER
        )
    """)
    conn.commit()
    conn.close()
create_table()

# =========================
# EMAIL OTP
# =========================
def send_otp_email(to_email, otp):
    msg = EmailMessage()
    msg.set_content(f"Your FileVerse OTP is {otp}. Valid for 5 minutes.")
    msg["Subject"] = "FileVerse OTP"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# =========================
# AUTH ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect("/home")
        
        flash("Invalid email or password")

    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            flash("Signup successful! Please login.")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Email already registered")

    return render_template("signup.html")

@app.route("/home")
@login_required
def home():
    return render_template("home.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================
# FORGOT PASSWORD
# =========================
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

        if user:
            otp = str(random.randint(100000, 999999))
            expiry = int(time.time()) + 300

            conn.execute(
                "UPDATE users SET otp=?, otp_expiry=? WHERE email=?",
                (otp, expiry, email)
            )
            conn.commit()
            conn.close()

            send_otp_email(email, otp)
            return redirect("/verify-otp")

        conn.close()
        flash("Email not registered")

    return render_template("forgot_password.html")

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        email = request.form["email"]
        otp = request.form["otp"]
        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()
        conn.close()

        if user and user["otp"] == otp and int(time.time()) < user["otp_expiry"]:
            return redirect(f"/reset-password/{email}")

        flash("Invalid or expired OTP")

    return render_template("verify_otp.html")


@app.route("/reset-password/<email>", methods=["GET", "POST"])
def reset_password(email):
    if request.method == "POST":
        new_password = generate_password_hash(request.form["password"])

        conn = get_db()
        conn.execute(
            "UPDATE users SET password=?, otp=NULL, otp_expiry=NULL WHERE email=?",
            (new_password, email)
        )
        conn.commit()
        conn.close()

        flash("Password reset successful")
        return redirect("/")

    return render_template("reset_password.html")

# =========================
# PDF COMPRESSION
# =========================
from werkzeug.utils import secure_filename
# =========================
# GHOSTSCRIPT DETECTION
# =========================
def get_ghostscript():
    if shutil.which("gswin64c"):
        return "gswin64c"   # Windows
    elif shutil.which("gs"):
        return "gs"         # Linux / Mac
    else:
        raise RuntimeError("Ghostscript not installed")

# =========================
# BEST COMPRESSION FUNCTION
# =========================
def compress_pdf_best(input_pdf, output_pdf):
    gs = get_ghostscript()
    qualities = ["screen", "ebook"]

    best_size = None
    best_file = None

    for q in qualities:
        temp_out = output_pdf.replace(".pdf", f"_{q}.pdf")

        cmd = [
            gs,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dSubsetFonts=true",
            "-dEmbedAllFonts=true",
            "-dDetectDuplicateImages=true",
            "-dDiscardAllComments=true",
            f"-dPDFSETTINGS=/{q}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={temp_out}",
            input_pdf
        ]

        subprocess.run(cmd, check=True)

        size = os.path.getsize(temp_out)
        if best_size is None or size < best_size:
            best_size = size
            best_file = temp_out

    shutil.move(best_file, output_pdf)

    # cleanup temp files
    for q in qualities:
        temp = output_pdf.replace(".pdf", f"_{q}.pdf")
        if os.path.exists(temp):
            os.remove(temp)

# =========================
# COMPRESS ROUTE (FINAL)
# =========================
@app.route("/compress", methods=["GET", "POST"])
@login_required
def compress():
    results = []

    if request.method == "POST":
        files = request.files.getlist("file")

        if not files or len(files) > 5:
            flash("You can upload a maximum of 5 PDF files")
            return render_template("compress.html", results=None)

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)

            output_name = filename.replace(".pdf", "_compressed.pdf")
            output_path = os.path.join(UPLOAD_FOLDER, output_name)

            # 1️⃣ Save original file
            file.save(input_path)

            # 🔥 REGISTER ORIGINAL FILE
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO uploads (filename, filepath, upload_time) VALUES (?, ?, ?)",
                (filename, input_path, int(time.time()))
            )
            conn.commit()
            conn.close()

            try:
                before_size = os.path.getsize(input_path)
                compress_pdf_best(input_path, output_path)
                after_size = os.path.getsize(output_path)

                saved = round((1 - after_size / before_size) * 100, 1)

                # 🔥 REGISTER COMPRESSED FILE
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute(
                    "INSERT INTO uploads (filename, filepath, upload_time) VALUES (?, ?, ?)",
                    (output_name, output_path, int(time.time()))
                )
                conn.commit()
                conn.close()

                results.append({
                    "name": filename,
                    "file": output_name,
                    "before": round(before_size / 1024, 2),
                    "after": round(after_size / 1024, 2),
                    "saved": saved
                })

            except Exception:
                flash(f"Compression failed for {filename}")

        return render_template("compress.html", results=results)

    # GET request
    return render_template("compress.html", results=None)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename,
        as_attachment=True
    )

from flask import send_file


# =========================
# GRAMMAR CHECK (IMPROVED)
# =========================
tool = language_tool_python.LanguageTool('en-US')
def highlight_errors(text, matches):
    highlighted = text
    offset = 0

    for match in matches:
        start = match.offset + offset
        end = start + match.error_length

        wrong = highlighted[start:end]

        css = "error-grammar"
        if match.rule_issue_type == "misspelling":
            css = "error-spelling"

        span = f'<span class="{css}" title="{match.message}">{wrong}</span>'

        highlighted = highlighted[:start] + span + highlighted[end:]
        offset += len(span) - len(wrong)

    return highlighted

def process_text(text):
    lines = text.splitlines()
    highlighted_lines = []
    corrected_lines = []

    for line in lines:
        # Skip headings (ALL CAPS)
        if line.strip().isupper():
            highlighted_lines.append(line)
            corrected_lines.append(line)
            continue

        matches = tool.check(line)
        highlighted_lines.append(highlight_errors(line, matches))
        corrected_once = language_tool_python.utils.correct(line, matches)
        second_pass = tool.check(corrected_once)
        corrected_final = language_tool_python.utils.correct(corrected_once, second_pass)

        corrected_lines.append(corrected_final)

    return "<br>".join(highlighted_lines), "<br>".join(corrected_lines)

@app.route("/grammar", methods=["GET", "POST"])
@login_required
def grammar():
    highlighted_text = None
    corrected_text = None
    error = None

    if request.method == "POST":
        raw_text = request.form.get("text", "").strip()
        file = request.files.get("file")

        if raw_text:
            highlighted_text, corrected_text = process_text(raw_text)

        elif file and file.filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
            highlighted_text, corrected_text = process_text(content)

        elif file and file.filename.endswith(".pdf"):
            content = extract_text_from_pdf(file)
            if content.strip():
                highlighted_text, corrected_text = process_text(content)
            else:
                error = "Unable to extract text from PDF."

        else:
            error = "Please enter text or upload a file."

    return render_template(
        "grammar.html",
        highlighted_text=highlighted_text,
        corrected_text=corrected_text,
        error=error
    )

# =========================
# SENSITIVE BLUR
# =========================
@app.route("/sensitive-blur")
@app.route("/blur")
@login_required
def sensitive_blur():
    return render_template("sensitive_blur.html")


@app.route("/upload-blur", methods=["POST"])
def upload_blur():
    global pages, edited_pages, uploaded_file_type

    # 🔥 CLEAN OLD FILES (VERY IMPORTANT)
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith("page_") or f.startswith("edited_") or f.endswith(".pdf"):
            os.remove(os.path.join(UPLOAD_DIR, f))

    pages = []
    edited_pages = {}

    file = request.files["file"]
    filename = file.filename.lower()

    # ---------- IMAGE SUPPORT ----------
    if filename.endswith((".png", ".jpg", ".jpeg")):
        uploaded_file_type = "image"

        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"pages": 0})

        path = os.path.join(UPLOAD_DIR, "page_0.png")
        img.save(path)
        pages.append(path)

        return jsonify({"pages": len(pages)})

    # ---------- PDF SUPPORT ----------
    elif filename.endswith(".pdf"):
        uploaded_file_type = "pdf"

        try:
            pdf = fitz.open(stream=file.read(), filetype="pdf")
        except Exception:
            return jsonify({"pages": 0})

        for i in range(len(pdf)):
            pix = pdf[i].get_pixmap(dpi=150)
            img_path = os.path.join(UPLOAD_DIR, f"page_{i}.png")
            pix.save(img_path)
            pages.append(img_path)

        pdf.close()
        return jsonify({"pages": len(pages)})

    return jsonify({"pages": 0})

@app.route("/get-page/<int:p>")
def get_page(p):
    if not pages:
        return "No pages loaded", 400

    if p < 0 or p >= len(pages):
        return "Page index out of range", 404

    return send_file(pages[p], mimetype="image/png")

@app.route("/save-blur", methods=["POST"])
def save_blur():
    data = request.json
    page = data["page"]
    image_data = data["image"].split(",")[1]

    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes))

    edited_path = os.path.join(UPLOAD_DIR, f"edited_{page}.png")
    img.save(edited_path)

    edited_pages[page] = edited_path
    return "", 204


@app.route("/download-final")
def download_final():
    global uploaded_file_type

    # ---------- IMAGE INPUT → IMAGE OUTPUT ----------
    if uploaded_file_type == "image":
        edited_img = os.path.join(UPLOAD_DIR, "edited_0.png")
        original_img = os.path.join(UPLOAD_DIR, "page_0.png")

        final_img = edited_img if os.path.exists(edited_img) else original_img

        if not os.path.exists(final_img):
            return "No image available", 400

        return send_file(final_img, as_attachment=True)

    # ---------- PDF INPUT → PDF OUTPUT ----------
    elif uploaded_file_type == "pdf":
        page_files = sorted(
            f for f in os.listdir(UPLOAD_DIR)
            if f.startswith("page_") and f.endswith(".png")
        )

        if not page_files:
            return "No file available", 400

        pdf_path = os.path.join(UPLOAD_DIR, "final_blurred.pdf")
        doc = fitz.open()

        for i, page_file in enumerate(page_files):
            edited_path = os.path.join(UPLOAD_DIR, f"edited_{i}.png")
            original_path = os.path.join(UPLOAD_DIR, page_file)

            img_path = edited_path if os.path.exists(edited_path) else original_path

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            page = doc.new_page(width=w, height=h)
            page.insert_image(page.rect, filename=img_path)

        doc.save(pdf_path)
        doc.close()

        return send_file(pdf_path, as_attachment=True)

    return "Unknown file type", 400


@app.route("/test")
def test():
    return "Flask routing works"

#========================
# COMPARISION
#========================
from difflib import SequenceMatcher

def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def compare_texts_similarity(text1, text2):
    matcher = SequenceMatcher(None, text1, text2)
    match_percent = round(matcher.ratio() * 100, 2)

    return {
        "match_percent": f"{match_percent}%",
        "preview1": f"<pre>{text1[:5000]}</pre>",
        "preview2": f"<pre>{text2[:5000]}</pre>",
        "excel": False
    }


@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare_files():
    result = None
    error = None

    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if not file1 or not file2:
            error = "Please upload both files"
            return render_template("compare.html", error=error)

        name1 = file1.filename.lower()
        name2 = file2.filename.lower()

        # ---------- TXT vs TXT ----------
        if name1.endswith(".txt") and name2.endswith(".txt"):
            text1 = file1.read().decode("utf-8", errors="ignore")
            text2 = file2.read().decode("utf-8", errors="ignore")
            result = compare_texts_similarity(text1, text2)

        # ---------- PDF vs PDF ----------
        elif name1.endswith(".pdf") and name2.endswith(".pdf"):
            text1 = extract_text_from_pdf(file1)
            text2 = extract_text_from_pdf(file2)
            result = compare_texts_similarity(text1, text2)

        # ---------- EXCEL vs EXCEL ----------
        elif name1.endswith((".xls", ".xlsx")) and name2.endswith((".xls", ".xlsx")):
            df1 = pd.read_excel(file1).fillna("").astype(str)
            df2 = pd.read_excel(file2).fillna("").astype(str)

            if list(df1.columns) != list(df2.columns):
                error = "Excel files must have the same column structure"
                return render_template("compare.html", error=error)

            df2_remaining = df2.copy()
            matched_rows = []

            for _, row in df1.iterrows():
                mask = (df2_remaining == row.values).all(axis=1)
                if mask.any():
                    matched_rows.append(row)
                    df2_remaining = df2_remaining.drop(mask.idxmax())

            matched_df = pd.DataFrame(matched_rows, columns=df1.columns)

            session["matched_excel"] = matched_df.to_dict(orient="records")
            session["matched_columns"] = list(df1.columns)

            # ✅ Highlight ONLY matched rows
            matched_set = set(tuple(row) for row in matched_df.values)

            def highlight_row(row):
                return [
                    "background-color:#fff176" if tuple(row.values) in matched_set else ""
                    for _ in row
                ]

            table1 = df1.style.apply(highlight_row, axis=1).to_html(index=False)
            table2 = df2.style.apply(highlight_row, axis=1).to_html(index=False)

            total_rows = len(df1)
            match_percent = round(
                (len(matched_df) / total_rows) * 100, 2
            ) if total_rows else 0

            result = {
                "excel": True,
                "match_percent": f"{match_percent}%",
                "matched_count": len(matched_df),
                "table1": table1,
                "table2": table2
            }

        else:
            error = "Please upload same file types (TXT, PDF, Excel)"

    return render_template("compare.html", result=result, error=error)

@app.route("/download-matched-pdf")
def download_matched_pdf():
    matched_rows = session.get("matched_excel")
    columns = session.get("matched_columns")

    if not matched_rows or not columns:
        return "No matched data available", 400

    pdf_path = os.path.join(UPLOAD_FOLDER, "matched_content.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Matched Content Report</b>", styles["Title"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    table_data = [columns]
    for row in matched_rows:
        table_data.append([str(row[col]) for col in columns])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)

@app.route("/download-matched-excel")
def download_matched_excel():
    matched_rows = session.get("matched_excel")
    columns = session.get("matched_columns")

    if not matched_rows or not columns:
        return "No matched data available", 400

    df = pd.DataFrame(matched_rows, columns=columns)

    excel_path = os.path.join(UPLOAD_FOLDER, "matched_content.xlsx")
    df.to_excel(excel_path, index=False)

    return send_file(excel_path, as_attachment=True)

#=====================
#extract
#======================
def build_tree(path, base):
    tree = {}
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        rel_path = os.path.relpath(full_path, base)

        if os.path.isdir(full_path):
            tree[item] = {
                "type": "dir",
                "path": rel_path,
                "children": build_tree(full_path, base)
            }
        else:
            tree[item] = {
                "type": "file",
                "path": rel_path
            }
    return tree

@app.route("/extract", methods=["GET", "POST"])
@login_required
def extract_zip():
    tree = None
    project_name = None

    if request.method == "POST":
        zip_file = request.files.get("zipfile")
        if zip_file and zip_file.filename.endswith(".zip"):
            project_name = zip_file.filename.replace(".zip", "").replace(" ", "_")
            extract_path = os.path.join("uploads", project_name)

            if os.path.exists(extract_path):
                shutil.rmtree(extract_path)

            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(extract_path)

            tree = build_tree(extract_path, extract_path)


    return render_template("extract.html", tree=tree, project_name=project_name)

@app.route("/download-extract/<project>")
def download_extracted_project(project):
    folder_path = os.path.join("uploads", project)

    if not os.path.exists(folder_path):
        return "Project not found", 404

    zip_base = os.path.join("uploads", project)
    shutil.make_archive(zip_base, "zip", folder_path)

    return send_file(zip_base + ".zip", as_attachment=True)

@app.route("/download-item/<project>/<path:item_path>")
def download_item(project, item_path):
    base_path = os.path.join("uploads", project)
    full_path = os.path.join(base_path, item_path)

    if not os.path.exists(full_path):
        return "File not found", 404

    # ✅ If FILE → download directly
    if os.path.isfile(full_path):
        return send_file(full_path, as_attachment=True)

    # ✅ If FOLDER → zip and download
    zip_name = item_path.replace("/", "_")
    zip_path = os.path.join("uploads", zip_name)

    shutil.make_archive(zip_path, "zip", full_path)

    return send_file(zip_path + ".zip", as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    if file:
        path = os.path.join("uploads", file.filename)
        file.save(path)

        upload_time = int(time.time())

        conn = sqlite3.connect("users.db")  # SAME DB you opened
        c = conn.cursor()
        c.execute(
            "INSERT INTO uploads (filename, filepath, upload_time) VALUES (?, ?, ?)",
            (file.filename, path, upload_time)
        )
        conn.commit()
        conn.close()

    return "File uploaded successfully"

@app.route('/test-upload')
def test_upload():
    import time, sqlite3, os

    path = os.path.join("uploads", "test.txt")
    with open(path, "w") as f:
        f.write("hello")

    upload_time = int(time.time())

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO uploads (filename, filepath, upload_time) VALUES (?, ?, ?)",
        ("test.txt", path, upload_time)
    )
    conn.commit()
    conn.close()

    return "Test file inserted"

def auto_delete_files():
    while True:
        now = int(time.time())

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute(
            "SELECT id, filepath FROM uploads WHERE ? - upload_time >= 300",
            (now,)
        )

        expired = c.fetchall()

        for file_id, path in expired:
            if os.path.exists(path):
                os.remove(path)
            c.execute("DELETE FROM uploads WHERE id = ?", (file_id,))

        conn.commit()
        conn.close()

        time.sleep(10)  # check every 5 minutes


threading.Thread(target=auto_delete_files, daemon=True).start()


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run()

