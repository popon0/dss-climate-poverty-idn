# dss/views/province.py
"""
Tampilan provinsi untuk Decision Support System (Streamlit dashboard).

Menampilkan:
- Tren emisi & penerimaan provinsi (dual-axis).
- Tren tarif adaptif provinsi.
- Movers emisi YoY untuk provinsi.
- Kontribusi provinsi terhadap total nasional (KPI, donut, bar ranking).
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact
from dashboard.charts import dual_axis

# === Warna konsisten untuk highlight provinsi ===
HIGHLIGHT = "#16a34a"   # hijau untuk provinsi terpilih
NEUTRAL   = "#94a3b8"   # abu untuk lainnya
ACCENT    = "#60a5fa"   # biru muda untuk "Top-10 lainnya"


def render_province(
    df: pd.DataFrame,
    df_year: pd.DataFrame,
    df_range: pd.DataFrame,
    year: int,
    TH: Theme,
    province: str,
    kpi: dict,
    tariff_rpkg: float,
    targets: dict
) -> None:
    """
    Render panel tampilan untuk 1 provinsi.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset lengkap (historis + prediksi).
    df_year : pd.DataFrame
        Subset data untuk tahun aktif.
    df_range : pd.DataFrame
        Subset data sesuai rentang tahun.
    year : int
        Tahun aktif.
    TH : Theme
        Objek tema visualisasi (light/dark).
    province : str
        Nama provinsi yang dipilih.
    kpi : dict
        KPI snapshot provinsi.
    tariff_rpkg : float
        Tarif rata-rata tertimbang (Rp/kg).
    targets : dict
        Target skala nasional/provinsi.
    """
    left, right = st.columns([1.6, 1])

    # === Left: Tren provinsi ===
    with left:
        st.subheader(f"Tren — {province}")
        agg = (
            df[df["province"] == province]
            .groupby("year", as_index=False)
            .agg({"Emisi (Ton)": "sum", "Penerimaan Negara (T)": "sum"})
        )
        fig_trend = dual_axis(agg, TH, "Emisi (Ton)", "Penerimaan Negara (T)")
        st.plotly_chart(
            compact(fig_trend, TH, h=400, legend_top=True),
            use_container_width=True
        )

        st.subheader("Tren Tarif (Adaptif) — Provinsi")
        grp = (
            df[df["province"] == province]
            .groupby("year", as_index=False)
            .apply(lambda d: pd.Series({
                "tarif_aw": (
                    (d["Tarif Pajak"].fillna(0) * d["Emisi (Ton)"].fillna(0)).sum()
                    / max(d["Emisi (Ton)"].fillna(0).sum(), 1e-9)
                )
            }))
            .reset_index(drop=True)
            .sort_values("year")
        )
        fig_tarif = px.line(
            grp, x="year", y="tarif_aw", markers=True,
            labels={"tarif_aw": "Rp/kg", "year": "Tahun"}
        )
        st.plotly_chart(compact(fig_tarif, TH, h=400), use_container_width=True)

    # === Right: Movers & kontribusi ===
    with right:
        st.subheader("Top Movers (Emisi YoY)")
        prev = (
            df[df["year"] == year - 1][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "prev"})
        )
        now = (
            df[df["year"] == year][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "now"})
        )
        yoy = now.merge(prev, on="province", how="inner").query("prev != 0")
        yoy["pct"] = (yoy["now"] - yoy["prev"]) / yoy["prev"] * 100
        sel = yoy.sort_values("pct", ascending=False).head(7).sort_values("pct")

        fig_mv = px.bar(
            sel, x="pct", y="province", orientation="h",
            color="pct", color_continuous_scale="Greens",
            labels={"pct": "%", "province": ""}
        )
        st.plotly_chart(compact(fig_mv, TH, h=240), use_container_width=True)

        st.subheader("Kontribusi & Posisi Emisi Provinsi")
        render_province_contribution(df, year, province, TH)


def render_province_contribution(
    df: pd.DataFrame,
    year: int,
    province: str,
    TH: Theme
) -> None:
    """
    Render kontribusi provinsi terhadap emisi nasional.

    Menampilkan:
    - KPI ringkas: peringkat, kontribusi nasional, porsi vs top-10.
    - Donut chart kontribusi provinsi dalam total nasional.
    - Bar chart ranking top-10 provinsi dengan highlight provinsi terpilih.
    """
    yr = df[df["year"] == year].copy()
    if yr.empty:
        st.info("Tidak ada data pada tahun ini.")
        return

    # --- agregasi nasional & ranking ---
    agg = (
        yr.groupby("province", as_index=False)["Emisi (Ton)"]
        .sum().sort_values("Emisi (Ton)", ascending=False).reset_index(drop=True)
    )
    total = float(agg["Emisi (Ton)"].sum())
    if total <= 0 or province not in agg["province"].values:
        st.info("Data emisi tidak tersedia atau provinsi tidak ditemukan.")
        return

    row = agg[agg["province"] == province].iloc[0]
    val = float(row["Emisi (Ton)"])
    rank = int(row.name) + 1
    nprov = len(agg)
    contrib = val / total * 100.0

    # --- top-10 ---
    top10 = agg.head(10).copy()
    top10_sum = float(top10["Emisi (Ton)"].sum())
    share_vs_top10 = (val / top10_sum * 100.0) if top10_sum else 0.0

    # === KPI ringkas ===
    k1, k2, k3 = st.columns(3)
    k1.metric("Peringkat Emisi", f"#{rank}", help=f"Dari {nprov} provinsi")
    k2.metric("Kontribusi Nasional", f"{contrib:.2f} %")
    k3.metric("Porsi vs Top-10", f"{share_vs_top10:.2f} %")

    # === Donut chart ===
    if province in top10["province"].values:
        top10_others = float(top10[top10["province"] != province]["Emisi (Ton)"].sum())
        others_rest = max(total - (val + top10_others), 0.0)
        ddf = pd.DataFrame({
            "Komponen": [province, "Top-10 lainnya", "Di luar Top-10"],
            "Emisi": [val, top10_others, others_rest],
        })
    else:
        others_rest = max(total - (val + top10_sum), 0.0)
        ddf = pd.DataFrame({
            "Komponen": [province, "Top-10 (tanpa provinsi)", "Provinsi lainnya"],
            "Emisi": [val, top10_sum, others_rest],
        })
    ddf["Persen"] = ddf["Emisi"] / total * 100.0

    col_map = {
        province: HIGHLIGHT,
        "Top-10 lainnya": ACCENT,
        "Top-10 (tanpa provinsi)": ACCENT,
        "Di luar Top-10": NEUTRAL,
        "Provinsi lainnya": NEUTRAL,
    }
    fig_pie = px.pie(
        ddf, names="Komponen", values="Persen",
        hole=0.6, color="Komponen", color_discrete_map=col_map
    )
    fig_pie.update_traces(
        texttemplate="%{label}<br>%{percent:.1%}",
        textposition="inside",
        pull=[0.08 if n == province else 0 for n in ddf["Komponen"]],
    )
    fig_pie.update_layout(
        title=f"Kontribusi Emisi {province} ({contrib:.2f}% dari total)",
        height=240, margin=dict(l=6, r=6, t=30, b=6), showlegend=False
    )
    st.plotly_chart(compact(fig_pie, TH, h=240, hide_cbar=True), use_container_width=True)

    # === Bar chart top-10 ===
    rank10 = agg.head(10).copy()
    if province not in rank10["province"].values:
        keep = rank10.tail(9)
        sel = agg[agg["province"] == province]
        rank10 = (
            pd.concat([keep, sel], ignore_index=True)
            .sort_values("Emisi (Ton)", ascending=True)
        )
    else:
        rank10 = rank10.sort_values("Emisi (Ton)", ascending=True)

    rank10["__col"] = rank10["province"].apply(
        lambda x: HIGHLIGHT if x == province else NEUTRAL
    )
    fig_bar = px.bar(
        rank10, x="Emisi (Ton)", y="province", orientation="h",
        color="__col", color_discrete_map="identity",
        labels={"province": "", "Emisi (Ton)": "Emisi (Ton)"}
    )
    fig_bar.update_layout(
        title="Posisi Emisi Top-10",
        height=260, margin=dict(l=6, r=6, t=30, b=6), showlegend=False
    )
    fig_bar.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.14, showarrow=False,
        text=f"#{rank} dari {nprov} provinsi • {contrib:.2f}% nasional • {share_vs_top10:.2f}% dari total Top-10",
        font=dict(size=11, color="#9aa4b2"),
    )
    st.plotly_chart(compact(fig_bar, TH, h=260, hide_cbar=True), use_container_width=True)
