from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from urllib.parse import urlparse, urljoin
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... (all imports remain the same)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Database Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
DB_URI = f"sqlite:///{DB_PATH}"

app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key"

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def initialize_database():
    with app.app_context():
        db.create_all()
        print("\u2705 Database initialized")

initialize_database()

# Load and prepare doctor data
doctor_df = pd.read_csv('doctor.csv')  # ensure this is loading properly

doctor_file = "doctor.csv"
if os.path.exists(doctor_file):
    df = pd.read_csv(doctor_file)
    df['Symptoms'] = df['Symptoms'].fillna('').str.lower().str.split(',')
    df['Location'] = df['Location '].fillna('').str.lower()
    df['SymptomText'] = df['Symptoms'].apply(lambda x: ' '.join(x))
    df['input_features'] = df['Location'] + " " + df['SymptomText']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['input_features'])
else:
    df = pd.DataFrame(columns=["Doctor Name", "Specialization", "Clinic/Hospital", "Contact", "Working Hours", 
                               "Ratings", "Latitude", "Longitude", "Symptoms", "Disease Expertise", "Location"])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([""])  # Avoid error if no data

appointments = []

def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/')
def home():
    return redirect(url_for('dashboard')) if current_user.is_authenticated else render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'GET':
        return render_template('signup.html')
    data = request.get_json()
    username, email, password = data.get('username'), data.get('email'), data.get('password')
    if not (username and email and password):
        return jsonify({'error': 'All fields are required'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400
    hashed_password = generate_password_hash(password)
    db.session.add(User(username=username, email=email, password=hashed_password))
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    data = request.get_json() if request.is_json else request.form
    email, password = data.get('email'), data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    login_user(user)
    next_page = request.args.get('next')
    if not is_safe_url(next_page):
        next_page = url_for("dashboard")
    return jsonify({'message': 'Login successful', 'redirect': next_page}), 200

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', email=current_user.email)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/form')
@login_required
def form():
    return render_template('form.html')

@app.route('/appointment')
@login_required
def appointment():
    return render_template('appointment.html')

@app.route('/feedback')
@login_required
def feedback():
    return render_template('feedback.html')



@app.route('/api/check_login')
def check_login():
    return jsonify({'logged_in': current_user.is_authenticated,
                    'username': current_user.username if current_user.is_authenticated else None})

@app.route('/api/doctors', methods=['POST'])
@login_required
def find_doctors():
    try:
        data = request.get_json()
        location = data.get("Location", "").strip().lower()
        symptoms = data.get("Symptoms", "").strip().lower()

        if not location or not symptoms:
            return jsonify({"success": False, "error": "Location and symptoms are required."}), 400

        df = pd.read_csv("doctor.csv")
        df.columns = df.columns.str.strip()  # 🧼 Fix column name issues

        print("Cleaned CSV Columns:", df.columns.tolist())  # ✅ DEBUG

        df['Location'] = df['Location'].fillna('').str.lower()
        df['Symptoms'] = df['Symptoms'].fillna('').str.lower()

        matched = df[
            df['Location'].str.contains(location, na=False) &
            df['Symptoms'].str.contains(symptoms, na=False)
        ]

        print("Matched rows:", matched.shape)

        if matched.empty:
            return jsonify({"success": True, "doctors": []})

        doctors = []
        for _, row in matched.iterrows():
          doctors.append({
            'name': row.get('Doctor Name', ''),
            'specialty': row.get('Specialization', ''),
            'clinic': row.get('Clinic/Hospital', ''),
            'contact': row.get('Contact', ''),
            'location': row.get('Location', ''),
            'ratings': row.get('Ratings', '')
        })
        return jsonify({"success": True, "doctors": doctors})

    except Exception as e:
        print("Error in /api/doctors:", e)
        return jsonify({"success": False, "error": str(e)})




@app.route('/api/specializations')
@login_required
def get_specializations():
    return jsonify({'success': True, 'specializations': df['Specialization'].dropna().unique().tolist()}), 200

@app.route('/api/symptoms')
@login_required
def get_symptoms():
    symptoms = []
    for sym_list in df['Symptoms']:
        symptoms.extend(sym_list if isinstance(sym_list, list) else [])
    return jsonify({'success': True, 'Symptoms': sorted(set(symptoms))}), 200

@app.route('/api/appointments', methods=['POST'])
@login_required
def book_appointment():
    try:
        data = request.get_json() if request.is_json else request.form
        appt = {
            'user': current_user.username,
            'name': data.get('name'),
            'doctor': data.get('doctor'),
            'date': data.get('date'),
            'time': data.get('time')
        }
        if not all(appt.values()):
            return jsonify({'success': False, 'error': 'All appointment fields are required'}), 400
        appointments.append(appt)
        return jsonify({'success': True, 'message': 'Appointment booked successfully'}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_appointments')
@login_required
def get_appointments():
    user_appts = [appt for appt in appointments if appt['user'] == current_user.username]
    return jsonify({'success': True, 'appointments': user_appts})

if __name__ == '__main__':
    app.run(debug=True, port=5000)