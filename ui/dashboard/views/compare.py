# dss/views/compare.py
"""
Tampilan perbandingan beberapa provinsi untuk Decision Support System.

Menampilkan:
- Tren multi-line emisi antar provinsi.
- Korelasi antar indikator (Pearson).
- Top movers emisi YoY.
- Perbandingan ringkas antar provinsi (dataframe).
- Radar profil (emisi, penerimaan, tarif, kemiskinan).
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.theme import Theme, compact


def render_compare(
    df: pd.DataFrame,
    df_year: pd.DataFrame,
    df_range: pd.DataFrame,
    year_range: tuple[int, int],
    year: int,
    TH: Theme,
    provinces: list[str],
    kpi: dict,
    tariff_rpkg: float,
    targets: dict
) -> None:
    """
    Render panel perbandingan antar provinsi.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset lengkap (historis + prediksi).
    df_year : pd.DataFrame
        Subset data untuk tahun aktif.
    df_range : pd.DataFrame
        Subset data sesuai rentang tahun.
    year_range : tuple[int, int]
        Range tahun untuk agregasi.
    year : int
        Tahun aktif.
    TH : Theme
        Objek tema visualisasi.
    provinces : list[str]
        Daftar provinsi yang dipilih untuk dibandingkan.
    kpi : dict
        KPI snapshot (tidak langsung dipakai di sini).
    tariff_rpkg : float
        Tarif rata-rata tertimbang (Rp/kg).
    targets : dict
        Target emisi, penerimaan, kemiskinan.
    """
    left, right = st.columns([1.6, 1])

    # === Left panel ===
    with left:
        head = ", ".join(provinces[:4]) + ("…" if len(provinces) > 4 else "")
        st.subheader(f"Tren — {head}")

        # Multi-line emisi historis/prediksi
        d = df[df["province"].isin(provinces)].copy()
        fig_line = px.line(d, x="year", y="Emisi (Ton)", color="province")
        st.plotly_chart(compact(fig_line, TH, h=400, legend_top=True), use_container_width=True)

        # Korelasi indikator
        st.subheader("Korelasi Indikator (Pearson)")
        scope = df[
            (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
            & (df["province"].isin(provinces))
        ].copy()
        agg = (
            scope.groupby("province", as_index=False)
            .agg({
                "Emisi (Ton)": "sum",
                "Penerimaan Negara (T)": "sum",
                "Kemiskinan (%)": "mean",
                "Tarif Pajak": "mean",
            })
        )
        corr = agg[
            ["Emisi (Ton)", "Penerimaan Negara (T)", "Kemiskinan (%)", "Tarif Pajak"]
        ].corr("pearson")
        fig_corr = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1,
            color_continuous_scale=TH.diverging, labels={"color": "ρ"}
        )
        st.plotly_chart(compact(fig_corr, TH, h=360, hide_cbar=False), use_container_width=True)

    # === Right panel ===
    with right:
        # Top movers
        st.subheader("Top Movers (Emisi YoY)")
        prev = (
            df[df["year"] == year - 1][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "prev"})
        )
        now = (
            df[df["year"] == year][["province", "Emisi (Ton)"]]
            .rename(columns={"Emisi (Ton)": "now"})
        )
        yoy = (
            now.merge(prev, on="province", how="inner")
            .query("prev != 0 and province in @provinces")
        )
        yoy["pct"] = (yoy["now"] - yoy["prev"]) / yoy["prev"] * 100
        sel = yoy.sort_values("pct", ascending=False).head(7).sort_values("pct")

        fig_mv = px.bar(
            sel, x="pct", y="province", orientation="h",
            color="pct", color_continuous_scale="Greens",
            labels={"pct": "%", "province": ""}
        )
        st.plotly_chart(compact(fig_mv, TH, h=260), use_container_width=True)

        # Perbandingan ringkas
        st.subheader("Perbandingan — Tahun Aktif")
        cmp = df[(df["year"] == year) & (df["province"].isin(provinces))].copy()
        grp = (
            cmp.groupby("province", as_index=False)
            .agg({
                "Emisi (Ton)": "sum",
                "Penerimaan Negara (T)": "sum",
                "Kemiskinan (%)": "mean"
            })
        )
        wtar = (
            cmp.groupby("province")
            .apply(lambda g: (
                (g["Tarif Pajak"] * g["Emisi (Ton)"]).sum()
                / max(g["Emisi (Ton)"].sum(), 1e-9)
            ))
            .reset_index(name="Tarif (Rp/kg)")
        )
        grp = grp.merge(wtar, on="province", how="left")
        total_nas = float(df[df["year"] == year]["Emisi (Ton)"].sum())
        grp["Share Nasional (%)"] = grp["Emisi (Ton)"] / max(total_nas, 1e-9) * 100
        grp["Rev Density (T per Mt)"] = (
            grp["Penerimaan Negara (T)"] * 1_000_000
        ) / grp["Emisi (Ton)"]

        st.dataframe(
            grp.rename(columns={
                "province": "Provinsi",
                "Penerimaan Negara (T)": "Penerimaan (T)"
            }),
            hide_index=True, use_container_width=True
        )

        # Radar profil
        st.subheader("Radar Profil — Emisi / Penerimaan / Tarif / Kemiskinan")

        def _norm(s: pd.Series) -> pd.Series:
            mn, mx = float(s.min()), float(s.max())
            return (s - mn) / (mx - mn) if mx > mn else s * 0

        rb = grp.copy()
        rb["E"] = _norm(rb["Emisi (Ton)"])
        rb["P"] = _norm(
            rb["Penerimaan (T)"] if "Penerimaan (T)" in rb else rb["Penerimaan Negara (T)"]
        )
        rb["T"] = _norm(rb["Tarif (Rp/kg)"])
        rb["K"] = 1 - _norm(rb["Kemiskinan (%)"])
        cats = ["Emisi", "Penerimaan", "Tarif", "Kemiskinan(↓)"]

        fig_r = go.Figure()
        for _, r in rb.iterrows():
            fig_r.add_trace(go.Scatterpolar(
                r=[r["E"], r["P"], r["T"], r["K"], r["E"]],
                theta=cats + ["Emisi"], name=r["province"], line=dict(width=2)
            ))
        fig_r.update_layout(
            height=320, margin=dict(l=0, r=0, t=25, b=25),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1]))
        )
        st.plotly_chart(compact(fig_r, TH, h=320), use_container_width=True)

    st.caption(
        "💡 Komparasi detail ANFIS vs Flat 30 (dalam **Revenue**) "
        "tersedia pada panel di bawah / modul *Perbandingan Revenue*."
    )
