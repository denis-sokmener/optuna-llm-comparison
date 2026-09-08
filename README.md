# LLM Destekli Optuna Hiperparametre Optimizasyonu

Bu proje, Scikit-learn RandomForestRegressor modeli üzerinde hiperparametre optimizasyonu (tuning) yaparken iki farklı Optuna yaklaşımını karşılaştırır:

1. **LLM Destekli Yöntem:** Google Gemini API kullanılarak veri seti istatistiklerine (satır/sütun sayısı) göre veri setine özel, mantıklı ve daraltılmış bir arama uzayı (search space) üretilir.
2. **Geleneksel Yöntem:** Literatürde genel kabul görmüş, standart ve çok geniş hiperparametre sınırları kullanılır.

Proje, LLM destekli aramanın benzer başarı (R2) skorlarını nasıl çok daha hafif, sığ ve canlı sisteme (production) uygun modellerle (düşük `max_depth` ve `n_estimators`) elde ettiğini kanıtlar.

## Proje Karşılaştırma Sonuçları

California Housing (~20.640 satır) regresyon veri seti üzerinde yapılan 10'ar denemelik Optuna optimizasyonu sonucunda aşağıdaki çıktılar elde edilmiştir:

* **LLM Destekli Model Skoru (R²):** 0.6582
  * *Bulunan Parametreler:* `{'max_depth': 13, 'n_estimators': 88}`

* **Geleneksel Model Skoru (R²):** 0.6587
  * *Bulunan Parametreler:* `{'max_depth': 49, 'n_estimators': 121}`

### Sonuç Analizi
Geleneksel yöntem sadece **0.0005** puanlık mikroskobik bir skor artışı için modelin karmaşıklığını devasa oranda artırmıştır (`max_depth: 49`). Bu durum modelin veriyi öğrenmek yerine ezberlemeye (overfitting) yatkın olduğunu gösterir. 

LLM destekli yöntem ise, Optuna'ya baştan mantıklı sınırlar çizerek aynı başarıyı çok daha sığ (13 derinlik) ve daha az ağaca (88 ağaç) sahip bir modelle bulmuştur. 
