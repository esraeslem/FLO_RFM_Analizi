[README.md](https://github.com/user-attachments/files/32284575/README.md)
# FLO Müşteri Segmentasyonu (RFM Analizi)

FLO'nun OmniChannel (hem online hem offline alışveriş yapan) müşterilerini **RFM
(Recency, Frequency, Monetary)** yöntemiyle segmentlere ayırıp, iki farklı
pazarlama senaryosu için hedef müşteri listeleri çıkaran uçtan uca bir analiz.

> Miuul AI Data Scientist Bootcamp — FLO RFM Case Study kapsamında
> hazırlanmıştır.

## İş Problemi

FLO, müşterilerini davranışlarına göre segmentlere ayırarak her segmente özel
pazarlama stratejileri belirlemek istiyor.

## Veri Seti

`data/flo_data_20k.csv` — 19.945 gözlem, 12 değişken.

| Değişken | Açıklama |
|---|---|
| `master_id` | Eşsiz müşteri numarası |
| `order_channel` | Kullanılan platform (Android, iOS, Desktop, Mobile) |
| `last_order_channel` | Son alışverişin yapıldığı kanal |
| `first_order_date` / `last_order_date` | İlk / son alışveriş tarihi |
| `last_order_date_online` / `_offline` | Kanal bazlı son alışveriş tarihi |
| `order_num_total_ever_online` / `_offline` | Kanal bazlı toplam sipariş sayısı |
| `customer_value_total_ever_online` / `_offline` | Kanal bazlı toplam harcama |
| `interested_in_categories_12` | Son 12 ayda ilgilenilen kategoriler |

## Proje Yapısı

```
├── data/
│   └── flo_data_20k.csv
├── src/
│   └── flo_rfm_analysis.py     # Uçtan uca analiz betiği
├── outputs/                    # Betik çalıştırılınca üretilen hedef listeler
├── docs/
│   └── FLO_RFM_Analizi.pdf     # Orijinal görev tanımı (Miuul)
├── requirements.txt
└── README.md
```

## Yöntem

1. **Veri hazırlama** — online/offline sipariş sayısı ve harcaması tek değişkende
   birleştirilir, tarih kolonları `datetime`'a çevrilir.
2. **RFM metrikleri** — analiz tarihi = son alışveriş tarihi + 2 gün.
   - `recency`: analiz tarihi − son alışveriş tarihi (gün)
   - `frequency`: toplam sipariş sayısı
   - `monetary`: toplam harcama
3. **Skorlama** — her metrik `qcut` ile 1-5 arasında skorlanır
   (`frequency`'deki çok sayıda eşit değer nedeniyle `rank(method="first")`
   kullanılır). `RF_SCORE = recency_score + frequency_score`.
4. **Segmentasyon** — `RF_SCORE`, aşağıdaki harita ile adlandırılmış
   segmentlere çevrilir:

   ```python
   seg_map = {
       r'[1-2][1-2]': 'hibernating',
       r'[1-2][3-4]': 'at_Risk',
       r'[1-2]5': 'cant_loose',
       r'3[1-2]': 'about_to_sleep',
       r'33': 'need_attention',
       r'[3-4][4-5]': 'loyal_customers',
       r'41': 'promising',
       r'51': 'new_customers',
       r'[4-5][2-3]': 'potential_loyalists',
       r'5[4-5]': 'champions'
   }
   ```
5. **Aksiyon (Görev 5)** — iki hedef müşteri listesi üretilir:
   - **Case A — Yeni kadın ayakkabı markası:** `champions` + `loyal_customers`
     segmentlerinden, `KADIN` kategorisiyle ilgilenen müşteriler →
     `outputs/yeni_marka_hedef_musteri_ids.csv`
   - **Case B — Erkek/Çocuk indirimi:** `cant_loose`, `hibernating`,
     `new_customers` segmentlerinden, `ERKEK` veya `COCUK` kategorisiyle
     ilgilenen müşteriler → `outputs/indirim_hedef_musteri_ids.csv`

   > Not: Görev metninin farklı sürümlerinde Case A için "ortalama 250 TL
   > üzeri harcama" koşulu geçebiliyor; resmi görev PDF'i bu koşulu
   > içermediği için varsayılan çözüm koşulsuz bırakılmıştır. Eklemek
   > isterseniz `export_new_brand_target_customers` içinde
   > `rfm["monetary"] > 250` filtresini bir satırla ekleyebilirsiniz.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python src/flo_rfm_analysis.py
```

Betik; veri keşfi çıktısını, kanal bazlı dağılımı, segment özet tablosunu
konsola yazar ve iki hedef müşteri listesini `outputs/` klasörüne kaydeder.

Pipeline'ı bir notebook veya başka bir betikten import ederek de
kullanabilirsiniz:

```python
from src.flo_rfm_analysis import run_flo_rfm_pipeline

df, rfm = run_flo_rfm_pipeline("data/flo_data_20k.csv", "outputs")
```

## Sonuç Özeti (bu veri seti için)

| Segment | Ort. Recency | Ort. Frequency | Ort. Monetary | Müşteri Sayısı |
|---|---:|---:|---:|---:|
| champions | 17.1 | 8.97 | 1410.71 | 1920 |
| loyal_customers | 82.6 | 8.36 | 1216.26 | 3375 |
| cant_loose | 235.2 | 10.72 | 1481.65 | 1194 |
| at_Risk | 242.3 | 4.47 | 648.33 | 3152 |
| need_attention | 113.0 | 3.74 | 553.44 | 806 |
| potential_loyalists | 36.9 | 3.31 | 533.74 | 2925 |
| hibernating | 247.4 | 2.39 | 362.58 | 3589 |
| about_to_sleep | 114.0 | 2.41 | 361.65 | 1643 |
| new_customers | 18.0 | 2.00 | 344.05 | 673 |
| promising | 58.7 | 2.00 | 334.15 | 668 |

- **Case A** (yeni marka hedef kitlesi): **2.487** müşteri
- **Case B** (indirim hedef kitlesi): **2.770** müşteri

## Lisans

Bu proje eğitim amaçlıdır; veri seti Miuul bootcamp materyaline aittir.
