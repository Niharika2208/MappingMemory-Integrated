import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import BytesIO

def parse_xml_to_dataframe(xml_file_path):
    # Load the XML file and parse it
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Prepare data extraction from XML
    data = []

    # Iterate through each child of the root (e.g., <Item> elements)
    for parent in root:
        parent_tag = parent.tag  # Tag name of the parent (e.g., "Item")
        row = {"Parent Tag": parent_tag}  # Start with the parent tag information

        # Add attributes of the parent tag
        row.update(parent.attrib)

        # Add child elements within the parent tag
        for child in parent:
            row[child.tag] = child.text

        data.append(row)

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    return df


def main():
    def display_converted_formats(data, file_format):
        # Flatten data for Excel format
        if file_format == "xml":
            df = data  # For XML, data is already a DataFrame
        else:
            df = pd.json_normalize(data, sep='_', max_level=1)

        # Excel buffer for download
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        excel_data = excel_buffer.getvalue()

        # Display options in Streamlit tabs
        json_tab, excel_tab = st.tabs(["JSON Format", "Excel Format"])

        with json_tab:
            if isinstance(data, list):
                # Display the first two items of the list for preview
                # st.json(data[:2])
                with st.expander("Show more"):
                    st.json(data)  # Full list
            elif isinstance(data, pd.DataFrame):
                # Convert DataFrame to a JSON-compatible format
                json_data = data.to_dict(orient='records')
                # st.json(json_data[:2])  # Display the first two rows
                with st.expander("Show more"):
                    st.json(json_data)  # Display all rows
            elif isinstance(data, dict):
                # Handle dictionaries
                # partial_data = dict(list(data.items())[:2])
                # st.json(partial_data)
                with st.expander("Show more"):
                    st.json(data)

        with excel_tab:
            st.dataframe(df, use_container_width=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "json", "xml"])

    if uploaded_file is not None:
        file_format = uploaded_file.name.split(".")[-1].lower()
        # Parse the uploaded file
        try:
            if file_format == "csv":
                data = pd.read_csv(uploaded_file).to_dict(orient="records")
            elif file_format == "xlsx":
                data = pd.read_excel(uploaded_file).to_dict(orient="records")
            elif file_format == "json":
                data = json.load(uploaded_file)
            elif file_format == "xml":
                data = parse_xml_to_dataframe(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return

            # Display the converted formats
            display_converted_formats(data, file_format)

        except Exception as e:
            st.error(f"Error processing the file: {e}")


if __name__ == "__main__":
    # st.set_page_config(page_title="File Converter with Parent Tag", page_icon="📄")
    main()
