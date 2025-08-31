# dss/views/extras.py
"""
Panel tambahan untuk DSS dashboard (Streamlit):
- Heatmap provinsi (indikator per tahun).
- Scenario optimizer untuk target 2030.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.theme import Theme, compact


def render_heatmap_and_optimizer(df: pd.DataFrame, TH: Theme) -> None:
    """
    Render panel Heatmap & Scenario Optimizer (2030).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset gabungan (historis + prediksi).
    TH : Theme
        Objek tema visualisasi (light/dark).
    """
    st.markdown("## Heatmap & Simulasi 2030")

    left, right = st.columns([2, 1], vertical_alignment="top")

    # === Heatmap provinsi ===
    with left:
        st.markdown("### 🔍 Heatmap Provinsi")
        indikator = st.selectbox(
            "Indikator Heatmap",
            ["Emisi (Ton)", "Kemiskinan (%)", "Penerimaan Negara (T)"],
            index=0, key="hm_indikator"
        )
        pvt = df.pivot_table(
            index="province", columns="year", values=indikator, aggfunc="sum"
        )
        fig = px.imshow(
            pvt, aspect="auto", color_continuous_scale="RdYlGn_r",
            labels={"color": indikator}
        )
        st.plotly_chart(compact(fig, TH, h=430, hide_cbar=False), use_container_width=True)

    # === Scenario Optimizer 2030 ===
    with right:
        st.markdown("### Scenario Optimizer (2030)")
        st.caption("Preset kebijakan → ubah slider → Hitung Dampak.")

        base_2030 = df[df["year"] == 2030].copy()
        baseline_em = base_2030["Emisi (Ton)"].sum()
        baseline_kem = base_2030["Kemiskinan (%)"].mean()
        baseline_rev = base_2030["Penerimaan Negara (T)"].sum()

        def eval_sken(pajak_up: float, emisi_down: float, kem_down: float) -> tuple[float, float, float]:
            """Hitung skenario baru berdasarkan perubahan % pajak, emisi, kemiskinan."""
            em_new = baseline_em * (1 - emisi_down / 100)
            kem_new = baseline_kem * (1 - kem_down / 100)
            rev_new = baseline_rev * (1 + pajak_up / 100) * (em_new / max(baseline_em, 1e-9))
            return em_new, kem_new, rev_new

        # --- Preset tombol cepat ---
        cA, cB, cC = st.columns(3)
        if cA.button("🌿 Green Push"):
            st.session_state.update({"_pajak": 10, "_emisi": 25, "_kem": 5})
        if cB.button("⚖️ Balanced"):
            st.session_state.update({"_pajak": 20, "_emisi": 15, "_kem": 8})
        if cC.button("💰 Revenue Max"):
            st.session_state.update({"_pajak": 40, "_emisi": 8, "_kem": 2})

        # --- Slider input ---
        tax_pct = st.slider("Kenaikan Pajak (%)", 0, 200, st.session_state.get("_pajak", 20), key="opt_tax")
        em_pct = st.slider("Penurunan Emisi (%)", 0, 100, st.session_state.get("_emisi", 10), key="opt_em")
        kem_pct = st.slider("Penurunan Kemiskinan (%)", 0, 100, st.session_state.get("_kem", 5), key="opt_kem")

        # --- Hitung dampak ---
        if st.button("▶️ Hitung Dampak", key="opt_run"):
            em_b, km_b, rv_b = eval_sken(tax_pct, em_pct, kem_pct)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Penerimaan 2030 (T)", f"{rv_b:.2f}",
                          delta=f"{(rv_b / baseline_rev - 1) * 100:+.1f}% vs baseline")
            with c2:
                st.metric("Emisi 2030 (Ton)", f"{em_b:,.0f}",
                          delta=f"{(em_b / baseline_em - 1) * 100:+.1f}% vs baseline")
            with c3:
                st.metric("Kemiskinan 2030 (%)", f"{km_b:.2f}",
                          delta=f"{(km_b / baseline_kem - 1) * 100:+.1f}% vs baseline")

        # === Sensitivitas penerimaan ===
        st.markdown("### 🔺 Sensitivitas Penerimaan (2030)")
        _, _, baseP = eval_sken(0, 0, 0)
        _, _, plus_tax = eval_sken(1, 0, 0)
        _, _, plus_em = eval_sken(0, 1, 0)

        drivers = pd.DataFrame({
            "Driver": ["Pajak (+1%)", "Penurunan Emisi (+1%)"],
            "Impact (T)": [plus_tax - baseP, plus_em - baseP]
        }).sort_values("Impact (T)")

        fig_drv = px.bar(
            drivers, x="Impact (T)", y="Driver", orientation="h",
            color="Impact (T)", color_continuous_scale="Blues",
            labels={"Driver": "", "Impact (T)": "Impact (T)"}
        )
        st.plotly_chart(compact(fig_drv, TH, h=260), use_container_width=True)
