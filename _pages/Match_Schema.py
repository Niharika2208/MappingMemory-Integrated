import streamlit as st
from st_cytoscape import cytoscape
import pandas as pd
import re

def main():
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        text = text.lower()
        text = text.replace("_", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters/numbers/spaces
        text = " ".join(text.split())  # collapse multiple spaces
        return text


    aml_dict = st.session_state.get("aml_dict", {})

    # Step 1: Extract SystemUnitClassLibs
    system_unit_libs = aml_dict.get("SystemUnitClassLib", [])
    lib_names = []
    lib_dict = {}

    for lib in system_unit_libs:
        name = lib.get("Name")
        if name:
            lib_names.append(name)
            lib_dict[name] = lib

    if not lib_names:
        st.warning("No SystemUnitClass Library found.")
        return

    selected_lib_name = st.selectbox("Select a SystemUnitClass Library:", lib_names)
    selected_lib = lib_dict[selected_lib_name]

    # Step 2: Extract SystemUnitClass entries
    system_unit_classes = selected_lib.get("SystemUnitClass", [])
    class_name_to_obj = {
        cls.get("Name"): cls for cls in system_unit_classes if cls.get("Name")
    }
    class_names = list(class_name_to_obj.keys())
    st.session_state["system_unit_class_names"] = class_names

    if not class_names:
        st.info("No SystemUnitClass entries found in this library.")
        return

    # Step 3: Build Cytoscape nodes
    elements = [{"data": {"id": class_name}} for class_name in class_names]

    # Step 4: Cytoscape graph with node selection enabled
    stylesheet = [
        {"selector": "node", "style": {"label": "data(id)", "width": 50, "height": 50}},
    ]

    st.subheader(f"SystemUnitClasses in '{selected_lib_name}'")
    selected_node = cytoscape(
        elements=elements,
        stylesheet=stylesheet,
        layout={"name": "grid"},
        height="500px",
        selection_type="single"
    )
    # st.write("DEBUG - selected_node:", selected_node)

    if selected_node:
        # DEBUG
        # st.write("DEBUG - selected_node:", selected_node)

        # Handle this structure: {"nodes": ["Process"], "edges": []}
        if isinstance(selected_node, dict) and "nodes" in selected_node:
            selected_nodes = selected_node["nodes"]
            if isinstance(selected_nodes, list) and len(selected_nodes) > 0:
                selected_class_name = selected_nodes[0]
            else:
                st.warning("No node selected.")
                return
        else:
            st.error("Unsupported selection format.")
            return

        st.session_state["system_unit_class_attributes"] = {}

        for class_name, class_obj in class_name_to_obj.items():
            attributes = class_obj.get("Attribute", [])
            attribute_names = [attr.get("Name", "") for attr in attributes if isinstance(attr, dict) and "Name" in attr]
            st.session_state["system_unit_class_attributes"][class_name] = attribute_names

        # Extract attribute names
        attribute_names = [attr.get("Name", "") for attr in attributes if isinstance(attr, dict) and "Name" in attr]

        # ✅ Store attribute names for cross-app use
        if "system_unit_class_attributes" not in st.session_state:
            st.session_state["system_unit_class_attributes"] = {}

        st.session_state["system_unit_class_attributes"][selected_class_name] = attribute_names

        # Display attributes if any
        if attribute_names:
            st.subheader(f"Attributes of '{selected_class_name}'")
            st.table(pd.DataFrame({"Attribute Name": attribute_names}))

            # --- NEW: Preprocess + store ---
            cleaned_names = [preprocess_text(a) for a in attribute_names]
            st.session_state["aml_attr_original"] = attribute_names
            st.session_state["aml_attr_clean"] = cleaned_names

            # Optional: store per class
            if "system_unit_class_attributes_clean" not in st.session_state:
                st.session_state["system_unit_class_attributes_clean"] = {}
            st.session_state["system_unit_class_attributes_clean"][selected_class_name] = cleaned_names

            # --- NEW DEBUG VIEW: show cleaned DataFrame ---
            #st.subheader(f"Cleaned Attributes for '{selected_class_name}' (Debug)")
            #clean_df = pd.DataFrame({
            #    "Original Attribute Name": attribute_names,
            #    "Cleaned Attribute Name": cleaned_names
            #})
            #st.dataframe(clean_df, use_container_width=True)

        else:
            st.info(f"No Attributes found under '{selected_class_name}'.")
