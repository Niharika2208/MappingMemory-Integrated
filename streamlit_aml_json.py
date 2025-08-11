import streamlit as st
from PIL import Image
import streamlit_antd_components as sac
from _pages import file_upload, check_conversion, optimize_storage, download
from _pages.Show_Common_Concept import main as Show_Common_Concept
# from _pages.Investigate_Raw_Data import main as Investigate_Raw_Data
# from _pages.Match_Schema import main as Match_Schema
from _pages.Investigate_and_Match import main as Investigate_and_Match
from _pages.Build_Mapping_Rules import main as Build_Mapping_Rules
from _pages.Match_Concept_to_Data import main as Match_Concept_to_Data
from _pages.Export_Data import main as Export_Data

# Load AML and header images
aml_image = Image.open('./aml_logo.png')
aml_header_logo = Image.open('./aml_header_logo.png')
aml_json_logo = Image.open('./aml_json_logo.png')

# Initialize Streamlit page configuration
st.set_page_config(page_title="AML-JSON Converter", page_icon=aml_image, initial_sidebar_state='expanded', layout='wide')

# Page header and title
st.image(aml_json_logo, width=500)

session_state_dict = {
    "aml_object": None,
    "raw_data": None,
    "json_size": None,
    "aml_dict": None,
    "optimized_yaml_string": None,
    "optimized_json_string": None,
    "optimized_dict": None,
    "aml_dict": None,
    "current_question": 1,
    "current_score": 0,
    "game_over": False,
    "schema_rules": {},
    "schema_attribute_rules": {},
    "found_schemas": [],
    "matched_entries": 0

}

for key, value in session_state_dict.items():
    if key not in st.session_state:
        st.session_state[key] = value

page_dict ={
    "Upload AML File": file_upload,
    "Upload JSON File": file_upload,
    "Check JSON conversion": check_conversion,
    "Optimize storage Size": optimize_storage,
    "Download JSON": download,
    "Common Concept": Show_Common_Concept,
    # "Investigate Raw Data": Investigate_Raw_Data,
    # "Match Schema": Match_Schema,
    "Investigate and Match": Investigate_and_Match,
    "Build Mapping Rules": Build_Mapping_Rules,
    "Match Concept to Data": Match_Concept_to_Data,
    "Export Data": Export_Data,
}

mode = sac.segmented(
    items=[
        sac.SegmentedItem(label='AML to JSON', icon='file-code'),
        sac.SegmentedItem(label='Data Mapping', icon="🧠"),
        sac.SegmentedItem(label='JSON to AML', icon='file-text'),
    ],
    label='Select Mode',
    align='center',
    use_container_width=True
)

if mode == 'AML to JSON':
    nav_items = ["Upload AML File", "Check JSON conversion", "Optimize storage Size", "Download JSON"]
elif mode == 'JSON to AML':
    nav_items = ["Upload JSON File"]
elif mode == 'Data Mapping':
    nav_items = ["Common Concept", "Investigate and Match", "Build Mapping Rules", "Match Concept to Data", "Export Data"]


# Navigation steps
nav_step = sac.steps(nav_items)

# Page navigation
page = page_dict[nav_step]()