import streamlit as st
import time
from streamlit_condition_tree import condition_tree, config_from_dataframe

def main():
    st.title("Build Mapping Rules")

    # Step 1: Load uploaded data
    if "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
        df = st.session_state["raw_df"]
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No uploaded file found. Please go to 'Investigate Raw Data' and upload a file first.")
        return

    # Step 2: Load schema mappings
    found_dict = st.session_state.get("found_schemas", {})
    found_schemas = list(dict.fromkeys([found_dict[i] for i in sorted(found_dict)]))

    if not found_schemas:
        st.warning("No schema mapping found. Please map entries in the previous step.")
        return

    # --- State setup for widget reset logic ---
    if "schema_rules" not in st.session_state:
        st.session_state["schema_rules"] = {}

    if "ct_key" not in st.session_state:
        st.session_state["ct_key"] = "ct_init"

    def on_schema_change():
        # Force a fresh instance of the condition_tree widget
        st.session_state["ct_key"] = f"ct_{st.session_state['selected_schema']}_{int(time.time())}"
        # Optional: clear any temporary preview variables you use
        st.session_state["current_rule_preview"] = None

    # Select current SystemUnitClass with change callback
    default_index = 0
    if "selected_schema" in st.session_state and st.session_state["selected_schema"] in found_schemas:
        default_index = found_schemas.index(st.session_state["selected_schema"])

    selected_schema = st.selectbox(
        "Select the concept for which you want to derive the rule:",
        found_schemas,
        index=default_index,
        key="selected_schema",
        on_change=on_schema_change,
    )

    # Step 3: Build condition tree (reset via ct_key when schema changes)
    config = config_from_dataframe(df)
    query_string = condition_tree(
        config,
        min_height=250,
        always_show_buttons=True,
        placeholder='Click "Add Rule" top right to build the mapping rule',
        key=st.session_state["ct_key"],   # <-- critical for resetting state
    )

    # Save rule locally in session (no DB)
    if st.button("Try Query", type="primary", use_container_width=True):
        if not selected_schema:
            st.error("Pick a SystemUnitClass.")
        elif not query_string or not str(query_string).strip():
            st.error("Enter a rule.")
        else:
            st.session_state["schema_rules"][selected_schema] = str(query_string).strip()
            st.success(f"Rule saved for {selected_schema}")

    # 🔍 Show ONLY the current schema's saved rule (hide all others)
    st.divider()
    current_rule = st.session_state.get("schema_rules", {}).get(selected_schema)
    if current_rule:
        st.write(f"Saved rule for **{selected_schema}**: `{current_rule}`")
    else:
        st.info(f"No rule yet for {selected_schema}")

    # Optional: compact “status tiles” per schema (without showing rules)
    st.divider()
    cols = st.columns(len(found_schemas))
    for i, schema in enumerate(found_schemas):
        if schema in st.session_state["schema_rules"]:
            cols[i].success(f"{schema}: rule set")
        else:
            cols[i].info(f"{schema}: no rule")
