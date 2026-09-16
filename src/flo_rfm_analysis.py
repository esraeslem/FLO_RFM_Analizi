"""
FLO Musteri Segmentasyonu - RFM Analizi
=========================================

Is Problemi
-----------
FLO musterilerini segmentlere ayirip bu segmentlere gore pazarlama stratejileri
belirlemek istiyor. Musteri davranislari tanimlanarak bu davranis oebeklenmelerine
gore gruplar (RFM segmentleri) olusturulacak.

Veri Seti
---------
2020-2021 yillarinda OmniChannel (hem online hem offline) alisveris yapan
FLO musterilerinin gecmis alisveris davranislarindan elde edilen bilgiler.

Kaynak: Miuul Data Science & Machine Learning Bootcamp - FLO RFM Case Study

Kullanim
--------
    python src/flo_rfm_analysis.py

Betik calistiginda tum pipeline (Gorev 1-6) sirasiyla yuruturulur ve
outputs/ klasorune iki hedef musteri listesi csv olarak yazilir.
"""

import os

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flo_data_20k.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


###############################################################
# GOREV 1: Veriyi Anlama ve Hazirlama
###############################################################

def check_df(dataframe: pd.DataFrame, head: int = 10) -> None:
    """Bir dataframe'in genel yapisini (boyut, tipler, bos deger, betimsel
    istatistik, ilk gozlemler) hizlica raporlar. Adim 2."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print(f"##################### Head({head}) #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)


def create_omnichannel_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adim 3 & 4: Omnichannel musteriler hem online hem offline alisveris
    yaptigindan, toplam siparis sayisi ve toplam harcamayi tek degiskende
    birlestirir; tarih iceren kolonlari datetime tipine cevirir.
    """
    dataframe = dataframe.copy()

    dataframe["order_num_total"] = (
        dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    )
    dataframe["customer_value_total"] = (
        dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    )

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    return dataframe


def data_prep(csv_path: str) -> pd.DataFrame:
    """Adim 8: Veri okuma + on hazirlik surecinin tamami tek fonksiyonda."""
    dataframe = pd.read_csv(csv_path)
    dataframe = create_omnichannel_metrics(dataframe)
    return dataframe


def explore_channels_and_top_customers(dataframe: pd.DataFrame) -> None:
    """Adim 5, 6, 7: Kanal bazli dagilim ve en degerli / en cok siparis veren
    ilk 10 musteri.
    """
    print("##################### Kanal Bazli Dagilim #####################")
    print(
        dataframe.groupby("order_channel").agg(
            musteri_sayisi=("master_id", "count"),
            toplam_siparis=("order_num_total", "sum"),
            toplam_harcama=("customer_value_total", "sum"),
        )
    )

    print("##################### En Karli Ilk 10 Musteri #####################")
    print(dataframe.sort_values("customer_value_total", ascending=False).head(10)[
        ["master_id", "customer_value_total"]
    ])

    print("##################### En Cok Siparis Veren Ilk 10 Musteri #####################")
    print(dataframe.sort_values("order_num_total", ascending=False).head(10)[
        ["master_id", "order_num_total"]
    ])


###############################################################
# GOREV 2: RFM Metriklerinin Hesaplanmasi
###############################################################

def create_rfm_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Analiz tarihini son alisveris tarihinden 2 gun sonrasi olarak alip
    her musteri icin recency, frequency, monetary metriklerini hesaplar.
    """
    analysis_date = dataframe["last_order_date"].max() + pd.Timedelta(days=2)

    rfm = dataframe.groupby("master_id").agg(
        recency=("last_order_date", lambda date: (analysis_date - date.max()).days),
        frequency=("order_num_total", "sum"),
        monetary=("customer_value_total", "sum"),
    )

    # frequency/monetary sifir ya da negatif olan gozlemler RFM analizinde anlamsizdir
    rfm = rfm[(rfm["frequency"] > 0) & (rfm["monetary"] > 0)]

    return rfm


###############################################################
# GOREV 3: RF ve RFM Skorlarinin Hesaplanmasi
###############################################################

def create_rf_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """recency, frequency, monetary metriklerini qcut ile 1-5 araliginda
    skorlara cevirir ve RF_SCORE (recency_score + frequency_score) uretir.

    Not: order_num_total degerinde cok sayida ayni deger (tie) bulundugundan
    frequency_score hesaplanirken qcut'un "Bin edges must be unique" hatasi
    vermemesi icin rank(method="first") kullanilir.
    """
    rfm = rfm.copy()

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["frequency_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
    rfm["RFM_SCORE"] = (
        rfm["recency_score"].astype(str)
        + rfm["frequency_score"].astype(str)
        + rfm["monetary_score"].astype(str)
    )

    return rfm


###############################################################
# GOREV 4: RF Skorlarinin Segment Olarak Tanimlanmasi
###############################################################

SEG_MAP = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions",
}


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """RF_SCORE'u yukaridaki seg_map ile aciklanabilir segment isimlerine cevirir."""
    rfm = rfm.copy()
    rfm["segment"] = rfm["RF_SCORE"].replace(SEG_MAP, regex=True)
    return rfm


###############################################################
# GOREV 5: Aksiyon Zamani!
###############################################################

def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Adim 1: Segmentlerin recency, frequency, monetary ortalama/count degerleri."""
    return rfm.groupby("segment").agg(
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        musteri_sayisi=("recency", "count"),
    ).sort_values("monetary", ascending=False)


def export_new_brand_target_customers(
    dataframe: pd.DataFrame, rfm: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """Case A: Yeni kadin ayakkabi markasi icin sadik (champions, loyal_customers)
    ve kadin kategorisinden alisveris yapan musterileri csv'ye kaydeder.
    """
    target_segments = ["champions", "loyal_customers"]
    target_customers = rfm[rfm["segment"].isin(target_segments)].index

    woman_category = dataframe[
        dataframe["interested_in_categories_12"].str.contains("KADIN", na=False)
    ]["master_id"]

    result = dataframe[
        dataframe["master_id"].isin(target_customers) & dataframe["master_id"].isin(woman_category)
    ][["master_id"]]

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, "yeni_marka_hedef_musteri_ids.csv"), index=False)
    print(f"[Case A] Hedef musteri sayisi: {len(result)}")
    return result


def export_discount_target_customers(
    dataframe: pd.DataFrame, rfm: pd.DataFrame, output_dir: str
) -> pd.DataFrame:
    """Case B: Erkek/Cocuk kategorisi indirimi icin kaybedilmemesi gereken
    (cant_loose), uykuda olan (hibernating) ve yeni gelen (new_customers)
    musterileri csv'ye kaydeder.
    """
    target_segments = ["cant_loose", "hibernating", "new_customers"]
    target_customers = rfm[rfm["segment"].isin(target_segments)].index

    men_kids_category = dataframe[
        dataframe["interested_in_categories_12"].str.contains("ERKEK|COCUK", na=False, regex=True)
    ]["master_id"]

    result = dataframe[
        dataframe["master_id"].isin(target_customers) & dataframe["master_id"].isin(men_kids_category)
    ][["master_id"]]

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, "indirim_hedef_musteri_ids.csv"), index=False)
    print(f"[Case B] Hedef musteri sayisi: {len(result)}")
    return result


###############################################################
# GOREV 6: Tum Sureci Fonksiyonlastirma
###############################################################

def run_flo_rfm_pipeline(csv_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
    """Ucdan uca RFM pipeline'i: veri okuma -> on hazirlik -> RFM metrikleri
    -> skorlar -> segmentler -> aksiyon listeleri.

    Returns
    -------
    dataframe : ham veri + turetilmis toplam kolonlar
    rfm       : musteri bazli recency/frequency/monetary/skor/segment tablosu
    """
    dataframe = data_prep(csv_path)
    rfm = create_rfm_metrics(dataframe)
    rfm = create_rf_scores(rfm)
    rfm = assign_segments(rfm)

    export_new_brand_target_customers(dataframe, rfm, output_dir)
    export_discount_target_customers(dataframe, rfm, output_dir)

    return dataframe, rfm


if __name__ == "__main__":
    df = data_prep(DATA_PATH)

    check_df(df)
    explore_channels_and_top_customers(df)

    rfm_df = create_rfm_metrics(df)
    rfm_df = create_rf_scores(rfm_df)
    rfm_df = assign_segments(rfm_df)

    print("##################### Segment Ozeti #####################")
    print(segment_summary(rfm_df))

    export_new_brand_target_customers(df, rfm_df, OUTPUT_DIR)
    export_discount_target_customers(df, rfm_df, OUTPUT_DIR)
