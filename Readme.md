📁 FileVerse – Secure File Utility Platform

FileVerse is a web-based file utility platform developed as a final-year project, providing multiple file management and analysis tools with secure authentication.

🚀 Features

🔐 User Authentication

Login / Signup

OTP-based Forgot Password

Secure session handling

📄 File Compression

Compress PDF files efficiently using Ghostscript

Supports multiple files (up to 5 at a time)

🧠 Grammar Checker

Detects grammar, spelling, and sentence structure errors

Provides highlighted mistakes and corrected output

Uses iterative correction for better accuracy

🔍 File Comparison

Compare TXT, PDF, and Excel files

Highlights matched and mismatched content

Download matched content as PDF or Excel

🗂 ZIP Extraction

Extract ZIP files

Browse extracted folder structure

Download individual files or entire folders

🕶 Sensitive Blur

Blur sensitive content in images and PDFs

Download final blurred file securely

⏱ Auto File Deletion

Uploaded files are automatically deleted after a fixed time for security

🛠 Technologies Used

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript

Database: SQLite

Libraries & Tools:

language-tool-python

PyMuPDF (fitz)

Pandas

Ghostscript

python-dotenv

🔐 Security Features

Session-based authentication

Protected routes using login_required

Browser cache disabled for secure navigation

Logout prevents access via back/forward buttons

Sensitive credentials managed using environment variables

📦 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/fileverse.git
cd fileverse

2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Create .env file
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

5️⃣ Run the application
python app.py


Open browser:

http://127.0.0.1:5000

📁 Project Structure
FileVerse/
├── app.py
├── templates/
├── static/
├── uploads/
├── .env
├── .gitignore
└── README.md

🎓 Academic Note

This project demonstrates:

Secure web authentication

File handling and processing

Rule-based NLP for grammar correction

Real-world web security practices

🧠 Future Improvements

Role-based access control

Cloud storage integration

Advanced AI-based grammar correction

Deployment on cloud platforms

👤 Author

Akhil P
Final Year Student
Department of Computer Science

🏁 License

This project is developed for academic purposes.