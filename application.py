import streamlit as st
import os
import os.path
from labeling_db import LabelDB
import tab_create_label

from datetime import datetime

# from labeling_libs.java_file_parser import JavaFileParser
from sidebar import Sidebar
from labeling_page import LabelingPage
# def save_annotation_to_db(st,label_db):
#     # Call the populate_tab function in tab_create_label.py to get checkbox values
#     checkbox_values = tab_create_label.get_checkbox_values(st)
#
#     print(checkbox_values)
#
#     # Iterate over the checkbox values and save each annotation to the database
#     for value in checkbox_values:
#         label_db.save_annotation(*value)

def authenticate_user(users_pwords, username, password):
    # Replace this with your actual authentication mechanism

    return username in users_pwords and password == users_pwords[username]

def main():

    if "all_users_passwords" not in st.session_state:
        st.session_state.all_users_passwords = {
            'admin': 'admin',
            'dipesh': 'dipesh',
            'user1': 'user1',
            'user2': 'user2',
            'user3': 'user3',
            'user4': 'user4',
            'user5': 'user5',
            'user6': 'user6',
            'user7': 'user7',
            'user8': 'user8',
            'user9': 'user9',
            'user10': 'user10'
        }

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if username and password:
            usrs_pwds = st.session_state.all_users_passwords
            st.session_state.authenticated = authenticate_user(usrs_pwds, username, password)
            if st.session_state.authenticated:
                st.session_state.user = username
                st.rerun()

    if not st.session_state.authenticated:
        return

    st.markdown(f'''Hello: <b>{st.session_state.user}</b>''', unsafe_allow_html=True)



    #session initializations

    if "date_today" not in st.session_state:
        st.session_state.date_today = datetime.now().strftime("%d-%m-%Y")

    if "json_dir" not in st.session_state:
        st.session_state.json_dir = "jsons"

    if "default_annotations" not in st.session_state:
        st.session_state.default_annotations = {"comment_label":{"relevant": 0, "quality": 1},
                                                "block_label":{"sufficient":0}
                                                }

    # initialize labeling data to None
    if "working_labeling_data" not in st.session_state:
        st.session_state.working_labeling_data = None

    if "working_file" not in st.session_state:
        st.session_state.working_file = ""

    st.title("Java Code Annotator")

    with st.sidebar:
        Sidebar.load_sidebar(st)

    LabelingPage.load_labeling_page(st)



if __name__ == "__main__":
    st.set_page_config(layout="wide")
    print("hello from main")
    main()
