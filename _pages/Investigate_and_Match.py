import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import BytesIO

def parse_xml_to_dataframe(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data = []
    for parent in root:
        row = {"Parent Tag": parent.tag}
        row.update(parent.attrib)
        for child in parent:
            row[child.tag] = child.text
        data.append(row)
    return pd.DataFrame(data)

def display_converted_formats(df):
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

    json_tab, excel_tab = st.tabs(["JSON Format", "Excel Format"])
    with json_tab:
        st.json(df.to_dict(orient="records"))
    with excel_tab:
        st.dataframe(df, use_container_width=True)

def main():
    st.subheader("Upload and Explore Raw Data")

    # Init session holders
    if "raw_file" not in st.session_state:
        st.session_state["raw_file"] = None
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json", "xml"])

    # Parse and cache
    if uploaded_file is not None:
        st.session_state["raw_file"] = uploaded_file
        file_format = uploaded_file.name.split(".")[-1].lower()
        st.session_state["source_file_type"] = file_format

        try:
            if file_format == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_format == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_format == "json":
                df = pd.json_normalize(json.load(uploaded_file))
            elif file_format == "xml":
                df = parse_xml_to_dataframe(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return

            st.session_state["raw_df"] = df
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            return

    # If we have data, render the two-column layout
    if st.session_state["raw_df"] is not None:
        df = st.session_state["raw_df"].copy()
        df.index = df.index + 1


        col_left, col_right = st.columns([3, 2])

        # LEFT: Data preview (tabs)
        with col_left:
            with st.expander("📊 Data Preview", expanded=False):
            # st.subheader("📊 Data Preview")
                display_converted_formats(df)

        # RIGHT: Schema Matching UI (inside expander)
        with col_right:
            with st.expander("Match Entries to SystemUnitClasses", expanded=False):
                st.info('Match each of the asset to their respective System Unit Classes', icon="ℹ️")
                class_options = st.session_state.get("system_unit_class_names", [])
                if not class_options:
                    st.warning("No SystemUnitClass names available. Please run 'Common Concept' first.")
                    return

                # Ensure found_schemas dict
                if "found_schemas" not in st.session_state or not isinstance(st.session_state["found_schemas"], dict):
                    st.session_state["found_schemas"] = {}

                num_rows = len(df)

                # Render in rows of 3 dropdowns
                for i in range(0, num_rows, 3):
                    row_chunk = df.iloc[i:i + 3]
                    cols = st.columns(len(row_chunk))
                    for j, (_, row) in enumerate(row_chunk.iterrows()):
                        idx = i + j
                        with cols[j]:
                            st.markdown(f"**Entry {idx + 1}**")
                            previous_value = st.session_state["found_schemas"].get(idx)
                            selected = st.selectbox(
                                label="SystemUnitClass",
                                options=class_options,
                                index=class_options.index(previous_value) if previous_value in class_options else 0,
                                key=f"class_select_{idx}"
                            )
                            st.session_state["found_schemas"][idx] = selected

                if st.button("✅ Confirm Mapping", use_container_width=True):
                    st.success("Your selections have been saved.")

    else:
        st.info("Upload a CSV/XLSX/JSON/XML file to begin.")
