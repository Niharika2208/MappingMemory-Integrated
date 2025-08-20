import streamlit as st
import streamlit_antd_components as sac
from sqlalchemy import text
from db_utils import insert_project

def check_conversion():
    nav_column, json_column = st.columns([3, 5])
    with nav_column:
        filter_item = sac.segmented(
            items=[
                sac.SegmentedItem(label='Whole file'),
                sac.SegmentedItem(label='Header Information'),
                sac.SegmentedItem(label='Instance Hierarchy'),
                sac.SegmentedItem(label='SystemUnitClass Library'),
                sac.SegmentedItem(label='RoleClass Library'),
                sac.SegmentedItem(label='Interface Class Library'),
                sac.SegmentedItem(label='AttributeType Library'),
            ],
            align='center',
            use_container_width=True,
            direction="vertical"
        )

    keys_of_interest_dict = {
        "Header Information": ["SchemaVersion", "FileName", "Revision", "RevisionDate", "OldVersion", "NewVersion",
                               "AuthorName", "AdditionalInformation"],
        "Instance Hierarchy": ["InstanceHierarchy"],
        "SystemUnitClass Library": ["SystemUnitClassLib"],
        "RoleClass Library": ["RoleClassLib"],
        "Interface Class Library": ["InterfaceClassLib"],
        "AttributeType Library": ["AttributeTypeLib"],
    }

    with json_column:
        aml_xsdict = st.session_state.get("aml_dict", {})

        st.info("Click the 3 dots to expand the file")

        # --- Extract OriginProjectID robustly ---
        def get_origin_info(data):
            """Extract OriginProjectID and OriginProjectTitle safely from AML dict."""
            result = {"OriginProjectID": None, "OriginProjectTitle": None}

            # 1) Direct path
            sdi = data.get("SourceDocumentInformation")
            if isinstance(sdi, dict):
                result["OriginProjectID"] = sdi.get("OriginProjectID")
                result["OriginProjectTitle"] = sdi.get("OriginProjectTitle")
                return result
            if isinstance(sdi, list):
                for item in sdi:
                    if isinstance(item, dict):
                        if "OriginProjectID" in item or "OriginProjectTitle" in item:
                            result["OriginProjectID"] = item.get("OriginProjectID")
                            result["OriginProjectTitle"] = item.get("OriginProjectTitle")
                            return result

            # 2) Fallback recursive search
            def _search(obj):
                if isinstance(obj, dict):
                    if "OriginProjectID" in obj or "OriginProjectTitle" in obj:
                        return {
                            "OriginProjectID": obj.get("OriginProjectID"),
                            "OriginProjectTitle": obj.get("OriginProjectTitle")
                        }
                    for v in obj.values():
                        found = _search(v)
                        if found:
                            return found
                elif isinstance(obj, list):
                    for el in obj:
                        found = _search(el)
                        if found:
                            return found
                return None

            found = _search(data)
            if found:
                result.update(found)

            return result

        # --- usage
        origin_info = get_origin_info(aml_xsdict)
        origin_id = origin_info.get("OriginProjectID")
        origin_title = origin_info.get("OriginProjectTitle")

        # check_conversion.py (after you compute origin_id and origin_title)
        if origin_id:
            st.session_state["origin_project_id"] = origin_id
            try:
                insert_project(str(origin_id).strip(),
                               str(origin_title).strip() if origin_title else None)
                # st.success(f"Project saved: {origin_id}")
            except Exception as e:
                st.error(f"Could not save project: {e}")

        if origin_title:
            st.session_state["origin_project_title"] = origin_title
            # st.info(f"Project Title: {origin_title}")
        else:
            st.warning("OriginProjectTitle not found in this file.")

        if filter_item == "Whole file":
            st.json(aml_xsdict, expanded=False)
        else:
            selected_keys = keys_of_interest_dict.get(filter_item)

            if selected_keys is None:
                st.error(f"❌ No keys defined for filter: '{filter_item}'")
                return

            partial_dict = {key: aml_xsdict[key] for key in selected_keys if key in aml_xsdict}
            st.json(partial_dict, expanded=False)
