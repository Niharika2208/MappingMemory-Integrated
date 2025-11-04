import streamlit as st
import pandas as pd

def main():
    st.title("Match Common Concept to Data")

    # Load user-uploaded data
    if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
        st.warning("No uploaded data found. Please start from 'Investigate Raw Data' and upload a file.")
        return

    df = st.session_state["raw_df"]
    found_dict = st.session_state.get("found_schemas", {})
    schema_rules = st.session_state.get("schema_rules", {})
    schema_attribute_rules = st.session_state.get("schema_attribute_rules", {})

    # Unique concepts, persisted for export
    found_schemas = list(dict.fromkeys([found_dict[i] for i in sorted(found_dict)]))
    st.session_state["unique_concepts"] = found_schemas

    if not found_schemas:
        st.warning("No schema mappings found. Please complete Step 2 before proceeding.")
        return

    # Select concept (schema)
    selected_schema = st.selectbox(
        "Select the concept to map attributes for:",
        found_schemas
    )

    # Apply filtering rule (if defined) to original dataframe
    try:
        query = schema_rules.get(selected_schema, "")
        filtered_df = df.query(query) if query else df
    except Exception as e:
        st.warning(f"Query failed: {e}")
        filtered_df = df
        # Keep all columns, even if all values are None
    cleaned_df = filtered_df

    # cleaned_df = filtered_df.dropna(axis=1, how='all')

    # 1) Get attributes from Show_Common_Concept.py
    concept_attributes = st.session_state.get("system_unit_class_attributes", {}).get(selected_schema, [])
    if not concept_attributes:
        st.warning(
            f"No attributes found for '{selected_schema}'. Please go to the 'Common Concept' app and select this node."
        )
        return

    # 2) Show filtered data table (always shown so you can see what you're mapping against)
    with st.expander("Filtered Data table", expanded=False):
        st.dataframe(cleaned_df, use_container_width=True)

    # 3) Toggle mapping direction
    st.divider()
    map_data_to_concept = st.checkbox("Map Data → Concept (instead of Concept → Data)", value=False)

    # Prepare common bits
    data_attribute_list = list(dict.fromkeys(cleaned_df.columns.tolist()))
    saved_attr_mapping = st.session_state.get("schema_attribute_rules", {}).get(selected_schema, {})
    saved_type_mapping = st.session_state.get("schema_attribute_types", {}).get(selected_schema, {})

    if not map_data_to_concept:
        st.subheader("Map Concept Attributes → Data Columns")

        # Build initial DF
        mapped_df = pd.DataFrame({
            "Concept Data": concept_attributes,
            "Data Attr": [saved_attr_mapping.get(attr, "") for attr in concept_attributes],
            "Type": [saved_type_mapping.get(attr, "string") for attr in concept_attributes]
        })

        # Keep old mapping for comparison
        old_mapping = saved_attr_mapping.copy()

        mapped_df = st.data_editor(
            mapped_df,
            use_container_width=True,
            key=f"editor_forward_{selected_schema}",
            column_config={
                "Data Attr": st.column_config.SelectboxColumn(
                    label="Choose the appropriate mapping",
                    options=data_attribute_list,
                ),
                "Type": st.column_config.SelectboxColumn(
                    label="Select data type",
                    options=["string", "int", "float", "boolean"],
                )
            },
            disabled=["Concept Data"]
        )

        # Collect success messages for changed mappings
        success_msgs = []
        for concept_attr, new_data_attr in zip(mapped_df["Concept Data"], mapped_df["Data Attr"]):
            if new_data_attr and old_mapping.get(concept_attr) != new_data_attr:
                success_msgs.append(f"'{concept_attr}' → '{new_data_attr}'")

        # Display in columns (3 per row)
        if success_msgs:
            num_cols = 5
            rows = [success_msgs[i:i + num_cols] for i in range(0, len(success_msgs), num_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col, msg in zip(cols, row):
                    col.success(msg)

        if st.button("✅ Validate Mapping", use_container_width=True, type="primary",
                     key=f"validate_forward_{selected_schema}"):
            attr_mapping = dict(zip(mapped_df["Concept Data"], mapped_df["Data Attr"]))
            type_mapping = dict(zip(mapped_df["Concept Data"], mapped_df["Type"]))

            st.session_state.setdefault("schema_attribute_rules", {})[selected_schema] = attr_mapping
            st.session_state.setdefault("schema_attribute_types", {})[selected_schema] = type_mapping
            st.success("Mapping saved (Concept → Data).")

    else:
        # ---------- Data → Concept ----------
        st.subheader("Map Data Columns → Concept Attributes")

        # Build reverse defaults from saved mapping (if any)
        reverse_saved = {}
        for c_attr, col in saved_attr_mapping.items():
            if col:
                reverse_saved[col] = c_attr

        # Also build reverse type mapping: column -> type
        reverse_types = {}
        for c_attr, col in saved_attr_mapping.items():
            if col:
                reverse_types[col] = saved_type_mapping.get(c_attr, "string")

        # Prepare rows = data columns; selection = concept attributes
        mapped_df_rev = pd.DataFrame({
            "Data Attr": data_attribute_list,
            "Concept Data": [reverse_saved.get(col, "") for col in data_attribute_list],
            "Type": [reverse_types.get(col, "string") for col in data_attribute_list],
        })

        # Store old mapping for comparison
        old_reverse_mapping = reverse_saved.copy()

        mapped_df_rev = st.data_editor(
            mapped_df_rev,
            use_container_width=True,
            key=f"editor_reverse_{selected_schema}",
            column_config={
                "Concept Data": st.column_config.SelectboxColumn(
                    label="Choose Concept Attribute",
                    options=concept_attributes,
                ),
                "Type": st.column_config.SelectboxColumn(
                    label="Select data type",
                    options=["string", "int", "float", "boolean"],
                ),
            },
            disabled=["Data Attr", "Mapped?"]
        )

        # Collect success messages for changed mappings
        success_msgs = []
        for data_col, new_concept_attr in zip(mapped_df_rev["Data Attr"], mapped_df_rev["Concept Data"]):
            if new_concept_attr and old_reverse_mapping.get(data_col) != new_concept_attr:
                success_msgs.append(f"'{data_col}' → '{new_concept_attr}'")

        # Display in columns (3 per row)
        if success_msgs:
            num_cols = 5
            rows = [success_msgs[i:i + num_cols] for i in range(0, len(success_msgs), num_cols)]
            for row in rows:
                cols = st.columns(len(row))
                for col, msg in zip(cols, row):
                    col.success(msg)

        if st.button("✅ Validate Mapping", use_container_width=True, type="primary",
                     key=f"validate_reverse_{selected_schema}"):
            # Build forward mapping from reverse table
            forward_map = {}
            type_mapping = {}
            for _, row in mapped_df_rev.iterrows():
                concept_attr = (row.get("Concept Data") or "").strip()
                data_col = row.get("Data Attr")
                if concept_attr:
                    forward_map[concept_attr] = data_col
                    type_mapping[concept_attr] = row.get("Type", "string")

            st.session_state.setdefault("schema_attribute_rules", {})[selected_schema] = forward_map
            st.session_state.setdefault("schema_attribute_types", {})[selected_schema] = type_mapping
            st.success("Mapping saved (Data → Concept).")
