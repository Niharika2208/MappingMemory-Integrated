import streamlit as st
import pandas as pd
from streamlit_condition_tree import condition_tree, config_from_dataframe


def main():
    st.title("Build Mapping Rules")

    # Step 1: Load uploaded data
    if "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
        df = st.session_state["raw_df"]
        # st.subheader("📄 Uploaded Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No uploaded file found. Please go to 'Investigate Raw Data' and upload a file first.")
        return

    # Step 2: Load schema mappings
    found_dict = st.session_state.get("found_schemas", {})
    found_schemas = list(dict.fromkeys([found_dict[i] for i in sorted(found_dict)]))
    # found_schemas = [found_dict[i] for i in sorted(found_dict)]

    if not found_schemas:
        st.warning("No schema mapping found. Please map entries in the previous step.")
        return

    selected_schema = st.selectbox(
        "Select the concept for which you want to derive the rule:",
        found_schemas
    )

    # st.subheader(f"🧠 Define rule for: {selected_schema}")

    # Step 3: Build condition tree
    config = config_from_dataframe(df)

    query_string = condition_tree(
        config,
        min_height=250,
        always_show_buttons=True,
        placeholder='Click "Add Rule" top right to build the mapping rule'
    )

    if "schema_rules" not in st.session_state:
        st.session_state["schema_rules"] = {}

    if st.button("Try Query", type="primary", use_container_width=True):
        st.session_state["schema_rules"][selected_schema] = query_string
        st.success(f"Rule saved for {selected_schema}")

    # 🔍 DEBUG print all saved rules
   # st.write("🔍 DEBUG - Saved Rules in Session State:")
    for schema, rule in st.session_state.get("schema_rules", {}).items():
        st.write(f"  • {schema}: `{rule}`")

    # Optional: Display existing rule validation
    st.divider()
    # st.subheader("📌 Status of Concept Rules")
    column_list = st.columns(len(found_schemas))

    for i, schema in enumerate(found_schemas):
        if schema not in st.session_state["schema_rules"]:
            column_list[i].info(f"No rule yet for {schema}")
        else:
            pass

           # column_list[i].success(f"Rule saved for {schema}")
