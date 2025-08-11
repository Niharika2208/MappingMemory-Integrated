import streamlit as st
import streamlit_antd_components as sac

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

        if filter_item == "Whole file":
            st.json(aml_xsdict, expanded=False)
        else:
            selected_keys = keys_of_interest_dict.get(filter_item)

            if selected_keys is None:
                st.error(f"❌ No keys defined for filter: '{filter_item}'")
                return

            partial_dict = {key: aml_xsdict[key] for key in selected_keys if key in aml_xsdict}
            st.json(partial_dict, expanded=False)
