import streamlit as st
from schemas import person_schema
from riddle_data import samples, sample_solutions

#st.title("Step 2: Match Schema")
def main():
    # Get the current sample based on session state
    selected_sample_key = st.session_state.get("current_question", 1)
    current_schema_solutions = list(sample_solutions[selected_sample_key].keys())
    schema = person_schema

    sample1_col, sample2_col, sample3_col = st.columns(3)

    # User selects the appropriate schema for each entry
    selected_schema_1 = sample1_col.selectbox(
        "Select schema for entry 1:",
        list(schema.keys())
    )

    selected_schema_2 = sample2_col.selectbox(
        "Select schema for entry 2:",
        list(schema.keys())
    )

    selected_schema_3 = sample3_col.selectbox(
        "Select schema for entry 3:",
        list(schema.keys())
    )

    # Remove duplicates while preserving order
    found_schemas = list(dict.fromkeys([selected_schema_1, selected_schema_2, selected_schema_3]))

    # Store in session state
    st.session_state["found_schemas"] = found_schemas

    # Validation logic
    if st.button("✅ Validate Selection", use_container_width=True, type="primary"):
        column_list = st.columns(len(found_schemas))
        for i, selected_schema in enumerate(found_schemas):
            try:
                if current_schema_solutions[i] == selected_schema:
                    column_list[i].success(f"Correct schema for Entry {i + 1}")
                else:
                    column_list[i].error(f"Incorrect schema for Entry {i + 1}")
            except IndexError:
                column_list[i].warning("Too many selected entries")