from pathlib import Path

import streamlit as st


IMAGE_DIR = Path(__file__).parent / "images"

st.title("Introduction to Data Mining")

left_column, right_column = st.columns([3, 2], gap="large", vertical_alignment="top")

with right_column:
    st.space("large")
    st.image(
        str(IMAGE_DIR / "crisp-dm.svg"),
        width="stretch",
        caption="CRISP-DM: Cross-Industry Standard Process for Data Mining"
    )

with left_column:
    tab_ph01, tab_ph02, tab_ph03, tab_ph04, tab_ph05, tab_ph06 = st.tabs([
            "01: Business Understanding", "02: Data Understanding", "03: Data Preparation", "04: Modeling", "05: Evaluation", "06: Deployment"
        ], 
    )
    
    with tab_ph01:
        st.subheader("Phase 1: Business Understanding")
        st.image(str(IMAGE_DIR / "crisp-dm-phase01.svg"), width="stretch", caption="Phase 1: Business Understanding")

    with tab_ph02:
        st.subheader("Phase 2: Data Understanding")
        st.image(str(IMAGE_DIR / "crisp-dm-phase02.svg"), width="stretch", caption="Phase 2: Data Understanding")
    
    with tab_ph03:
        st.subheader("Phase 3: Data Preparation")
        st.image(str(IMAGE_DIR / "crisp-dm-phase03.svg"), width="stretch", caption="Phase 3: Data Preparation")
    
    with tab_ph04:
        st.subheader("Phase 4: Modeling")
        st.image(str(IMAGE_DIR / "crisp-dm-phase04.svg"), width="stretch", caption="Phase 4: Modeling")

    with tab_ph05:
        st.subheader("Phase 5: Evaluation")
        st.image(str(IMAGE_DIR / "crisp-dm-phase05.svg"), width="stretch", caption="Phase 5: Evaluation")
    
    with tab_ph06:
        st.subheader("Phase 6: Deployment")
        st.image(str(IMAGE_DIR / "crisp-dm-phase06.svg"), width="stretch", caption="Phase 6: Deployment")
