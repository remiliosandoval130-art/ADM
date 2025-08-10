from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import os  # <<--- Importar aquí, no en la línea de la URI

app = Flask(__name__)
app.secret_key = "clave_secreta_super_segura"

# Usar Postgres desde la variable de entorno
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ------------------ MODELO DE USUARIO ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

with app.app_context():
    db.create_all()

# ------------------ LISTA DE SÍNTOMAS (expandida) ------------------
SYMPTOMS = [
    "fiebre","tos","dolor_garganta","congestion","dolor_cabeza","dolor_muscular",
    "fatiga","nauseas","vomito","diarrea","dolor_abdominal","perdida_olfato",
    "erupcion","presion_alta","glucosa_alta","saturacion_baja"
]

# Mapeo amistoso para UI
SYMPTOMS_LABELS = {
    "fiebre":"Fiebre",
    "tos":"Tos",
    "dolor_garganta":"Dolor de garganta",
    "congestion":"Congestión nasal",
    "dolor_cabeza":"Dolor de cabeza",
    "dolor_muscular":"Dolor muscular",
    "fatiga":"Fatiga",
    "nauseas":"Náuseas",
    "vomito":"Vómito",
    "diarrea":"Diarrea",
    "dolor_abdominal":"Dolor abdominal",
    "perdida_olfato":"Pérdida de olfato",
    "erupcion":"Erupción cutánea",
    "presion_alta":"Presión alta (reportada)",
    "glucosa_alta":"Glucosa alta (reportada)",
    "saturacion_baja":"Saturación de O₂ baja (reportada)"
}

# ------------------ "IA" DEMO: entrenamiento sintético ------------------
# Etiquetas
CLASSES = np.array([
    "Infección respiratoria",
    "Gastroenteritis",
    "Migraña/Tensional",
    "COVID-similar",
    "Dermatitis/Alérgica",
    "Hipertensión descontrolada",
    "Hiperglucemia",
    "Hipoxemia (revisar)"
])

# Creamos un dataset binario sintético (solo para demo)
# Cada fila = vector de 16 síntomas (0/1). Esto NO es diagnóstico real.
rng = np.random.default_rng(42)
X_train = rng.integers(0, 2, size=(64, len(SYMPTOMS)))
y_train = rng.integers(0, len(CLASSES), size=(64,))
clf = BernoulliNB()
clf.fit(X_train, y_train)

def infer_top(scores):
    # Devuelve lista de (clase, prob%) ordenada desc
    order = np.argsort(scores)[::-1]
    prob = (scores[order] / scores[order].sum()) if scores.sum() > 0 else np.ones_like(scores[order])/len(scores)
    prob = (prob * 100).round().astype(int)
    out = [(CLASSES[i], int(p)) for i, p in zip(order, prob)]
    return out[:3]

def triage_level(sym_vector):
    # Semáforo de prioridad simple basado en red flags
    red_flags = 0
    idx = {k:i for i,k in enumerate(SYMPTOMS)}
    # Señales críticas reportadas
    if sym_vector[idx["saturacion_baja"]] == 1: red_flags += 2
    if sym_vector[idx["presion_alta"]] == 1: red_flags += 1
    if sym_vector[idx["glucosa_alta"]] == 1: red_flags += 1
    # Fiebre + dolor_cabeza + fatiga = sumar 1
    if sum(sym_vector[[idx["fiebre"], idx["dolor_cabeza"], idx["fatiga"]]]) >= 2:
        red_flags += 1
    if red_flags >= 3: return ("ALTA", "Atención prioritaria en las próximas horas.")
    if red_flags == 2: return ("MEDIA", "Vigilar 24 h, considerar consulta.")
    return ("BAJA", "Autocuidado y seguimiento de síntomas.")

# ------------------ RUTAS ------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("form"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session["user_id"] = user.id
            return redirect(url_for("form"))
        return render_template("login.html", error="Credenciales incorrectas")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        password2 = request.form.get("password2","")
        if not email or not password:
            return render_template("register.html", error="Completa todos los campos")
        if password != password2:
            return render_template("register.html", error="Las contraseñas no coinciden")
        if User.query.filter_by(email=email).first():
            return render_template("register.html", error="Ese correo ya existe")
        db.session.add(User(email=email, password=password))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/form", methods=["GET","POST"])
def form():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        nombre = request.form.get("nombre","")
        edad = request.form.get("edad","")
        identidad = request.form.get("identidad","")
        antecedentes = request.form.get("antecedentes","")

        # Construir vector binario de síntomas
        x = np.zeros(len(SYMPTOMS), dtype=int)
        for i, key in enumerate(SYMPTOMS):
            if request.form.get(key):
                x[i] = 1

        scores = clf.predict_log_proba([x])[0]  # log-probs
        scores = np.exp(scores)                 # pasar a "probabilidad" (no calibrada)
        ranking = infer_top(scores)

        nivel, msg = triage_level(x)
        seleccionados = [SYMPTOMS_LABELS[k] for i,k in enumerate(SYMPTOMS) if x[i]==1]

        return render_template(
            "result.html",
            nombre=nombre, edad=edad, identidad=identidad,
            antecedentes=antecedentes, ranking=ranking,
            nivel=nivel, mensaje_triaje=msg,
            seleccionados=seleccionados
        )
    return render_template("form.html", symptoms=SYMPTOMS, labels=SYMPTOMS_LABELS)

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

# API pequeña para UI dinámica (conteo de síntomas marcados)
@app.route("/api/symcount", methods=["POST"])
def symcount():
    data = request.get_json(force=True)
    count = sum(1 for v in data.values() if v)
    return jsonify({"count": count})

if __name__ == "__main__":
    app.run(debug=False)
# --- al final de tus modelos ---
with app.app_context():
    db.create_all()  # crea las tablas si no existen
