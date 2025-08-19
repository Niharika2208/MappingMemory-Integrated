import streamlit as st
import json
from model import Caexfile              # NEW
from utils import parser_definition, json_unabbreviate  # NEW

def main():
    aml_dict = st.session_state.get("aml_dict", {})
    schema_rules = st.session_state.get("schema_rules", {})
    schema_attr_mappings = st.session_state.get("schema_attribute_rules", {})
    schema_attr_types = st.session_state.get("schema_attribute_types", {})
    found_schemas = st.session_state.get("unique_concepts", [])

    source_file = st.session_state.get("raw_file")
    source_file_type = source_file.name.split(".")[-1] if source_file else "csv"

    if "lib_toolnames" not in st.session_state:
        st.session_state["lib_toolnames"] = {}

    concept_mapping = []

    # Prepare list of libraries with valid names
    libs = [lib for lib in aml_dict.get("SystemUnitClassLib", []) if lib.get("Name")]

    # 🟦 Show toolname input per library in columns
    cols_per_row = 4
    st.subheader("Tool names:")
    st.info('Enter the Tool name and press "Enter" ', icon="ℹ️")
    for i in range(0, len(libs), cols_per_row):
        row_libs = libs[i:i + cols_per_row]
        columns = st.columns(len(row_libs))

        for col, lib in zip(columns, row_libs):
            lib_name = lib["Name"]
            previous_toolname = st.session_state["lib_toolnames"].get(lib_name, "")
            with col:
                toolname = st.text_input(f"'{lib_name}'", value=previous_toolname, key=f"toolname_{lib_name}")
                st.session_state["lib_toolnames"][lib_name] = toolname

    # 🔁 Process all SUCs with assigned toolnames
    for lib in libs:
        lib_name = lib["Name"]
        toolname = st.session_state["lib_toolnames"].get(lib_name, "")

        if not toolname.strip():
            continue

        for system_unit in lib.get("SystemUnitClass", []):
            concept = system_unit.get("Name")
            if not concept or concept not in schema_rules:
                continue

            rule = schema_rules[concept]
            attr_map = schema_attr_mappings.get(concept, {})
            type_map = schema_attr_types.get(concept, {})

            attribute_mapping = []
            for attr_name, raw_col in attr_map.items():
                attribute_mapping.append({
                    "suc_attr_name": attr_name,
                    "raw_data_name": raw_col,
                    "attribute_type": "None",
                    "type": type_map.get(attr_name, "string")
                })

            clean_entry = [{
                "suc_name": concept,
                "source_file_type": source_file_type,
                "toolname": toolname,
                "rule": rule.replace('"', "'"),
                "attribute_mapping": attribute_mapping
            }]

            concept_mapping.append(clean_entry)

            # Inject into AdditionalInformation
            json_string = json.dumps(clean_entry, separators=(",", ":"))
            system_unit["AdditionalInformation"] = [{"__root__": f"ConceptMapping: {json_string}"}]

    # ✅ Final Output
    final_json = {"ConceptMapping": concept_mapping}

    # st.subheader("📦 AML Dict Updated with Additional_Information")
    st.json(aml_dict)

    # NEW: Convert enriched dict back to AML and provide download
    if aml_dict:
        try:
            context, parser, json_parser, xml_serializer, json_serializer = parser_definition()
            full_json = json_unabbreviate(aml_dict)
            xml_obj: Caexfile = json_parser.bind_dataclass(full_json, Caexfile)
            xml_string = xml_serializer.render(xml_obj)

            st.download_button(
                "⬇️ Download AML (.aml)",
                data=xml_string,
                file_name=f"{st.session_state.get('file_name','aml_export')}.aml",
                mime="text/xml",
                use_container_width=True
            )
            st.success("AML file with ConceptMappings exported successfully ✅")
        except Exception as e:
            st.error(f"Failed to export AML: {e}")
