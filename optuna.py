import os
import json
import optuna
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings

# Optuna loglarını kapatalım
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ==========================================
# GÜVENLİ API YAPILANDIRMASI
# ==========================================
# .env dosyasındaki değişkenleri sisteme yükler
load_dotenv()

# API anahtarını ortam değişkenlerinden çeker
API_KEY = os.getenv("GEMINI_API_KEY")

# Anahtar bulunamazsa kodu durdur ve hata fırlat
if not API_KEY:
    raise ValueError("API anahtarı bulunamadı! Lütfen .env dosyanızı kontrol edin.")

genai.configure(api_key=API_KEY)

# ==========================================
# VERİ SETİ VE LLM HAZIRLIĞI
# ==========================================
# Veri Seti Yükleme (California Housing)
X, y = fetch_california_housing(return_X_y=True)
n_samples, n_features = X.shape

# LLM'e Gönderilecek Komut
prompt = f"""
Elimde {n_samples} satır ve {n_features} sütundan oluşan bir regresyon veri seti var (Ev fiyatı tahmini). 
Scikit-learn RandomForestRegressor için Optuna hiperparametre aralıkları belirlemek istiyorum.
Lütfen bana SADECE aşağıdaki yapıda, bu veri setinin boyutlarına uygun bir JSON döndür. Ekstra metin yazma:
{{
  "max_depth": {{"low": 0, "high": 0}},
  "n_estimators": {{"low": 0, "high": 0}}
}}
"""

print(f"1. AŞAMA: LLM'den {n_samples} satırlık veri için hiperparametre uzayı isteniyor...")
llm_model = genai.GenerativeModel('gemini-3.6-flash')
response = llm_model.generate_content(
    prompt,
    generation_config={"response_mime_type": "application/json"}
)

search_space = json.loads(response.text)
print("LLM'in Belirlediği Arama Uzayı:")
print(json.dumps(search_space, indent=2))
print("-" * 40)

# ==========================================
# YÖNTEM 1: LLM Destekli Optuna
# ==========================================
def objective_llm(trial):
    max_depth = trial.suggest_int("max_depth", search_space["max_depth"]["low"], search_space["max_depth"]["high"])
    n_estimators = trial.suggest_int("n_estimators", search_space["n_estimators"]["low"], search_space["n_estimators"]["high"])
    
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1)
    return cross_val_score(model, X, y, cv=3).mean()

print("2. AŞAMA: LLM sınırlarıyla Optuna çalışıyor (10 deneme)...")
study_llm = optuna.create_study(direction="maximize")
study_llm.optimize(objective_llm, n_trials=10) 

# ==========================================
# YÖNTEM 2: Geleneksel Optuna
# ==========================================
def objective_traditional(trial):
    max_depth = trial.suggest_int("max_depth", 2, 50)
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1)
    return cross_val_score(model, X, y, cv=3).mean()

print("\n3. AŞAMA: Geleneksel geniş sınırlarla Optuna çalışıyor (10 deneme)...")
study_traditional = optuna.create_study(direction="maximize")
study_traditional.optimize(objective_traditional, n_trials=10) 

# ==========================================
# KARŞILAŞTIRMA RAPORU
# ==========================================
print("\n" + "=" * 40)
print("KARŞILAŞTIRMA SONUCU (Skorlar R2'dir - 1.0 en mükemmel)")
print("=" * 40)
print(f"LLM Destekli Skor: {study_llm.best_value:.4f}")
print(f"LLM Parametreleri: {study_llm.best_params}\n")

print(f"Geleneksel Skor:   {study_traditional.best_value:.4f}")
print(f"Geleneksel Params: {study_traditional.best_params}")
print("=" * 40)
