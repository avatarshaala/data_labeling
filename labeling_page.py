
from labeling_libs.java_to_json import JavaToJSON
from labeling_libs.annotator_ui_component import ClickableGrid
# from streamlit_float import *
from streamlit_modal import Modal
import pandas as pd
class LabelingPage:

    @staticmethod
    def __save_annotation(st, create_new=None):

        data = st.session_state.working_labeling_data
        
        json_file = st.session_state.working_file
        json_dir = st.session_state.json_dir
        user = st.session_state.user
        date = st.session_state.date_today
        JavaToJSON.save_data_to_json(data, json_file, json_dir, user, date, create_new=create_new)

    @staticmethod
    def __separate_blocks(st, data):

        # data = st.session_state.labeling_data

        block_types = {
            "begin_header": "header",
            "header_comment": "header",
            "end_header": "header",
            "begin_comment": "comment",
            "comment": "comment",
            "end_comment": "comment",
            "code_body": "code",
            "": "blank_space"
        }


        blocks = []
        block = []
        running_block = ""
        mycode = []
        for line_num, line_data in data['lines'].items():

            type = line_data['type']
            print(f"{line_num} {block_types[type]} {running_block}")
            mycode.append(f"{line_num} {line_data['content']}")

            # if block type of the line is different from running block, append the block to blocks
            if (block_types[type] != running_block) and block:
                # print("____________________________")
                blocks.append((running_block, block))

                #clear the block and update block and running_block with the new one
                block = []
                block.append({f"{line_num}": line_data})
                running_block = block_types[type]

            else:
                # print(f"{line_num}: {type}")
                block.append({f"{line_num}":line_data})
                running_block = block_types[type]

        # append the last block if it is not appended inside the loop
        if block:
            blocks.append((running_block, block))
        return blocks, mycode


    @staticmethod
    def __show_header(st, data):
        top_col1, top_col2 = st.columns(2)
        with top_col1:
            c1, c2 = st.columns(2)
            with c1:
                save_clicked = st.button("Save Annotation")
                if save_clicked:
                    LabelingPage.__save_annotation(st, create_new=None)
                    st.write(f"{st.session_state.working_file} saved")
                with c2:
                    st.markdown(
                        f"Working File: <b>{st.session_state.working_file}</b>",
                        unsafe_allow_html=True)

        with top_col2:
            st.write("Click to see data in JSON")
            st.json(data, expanded=False)

        st.divider()

    @staticmethod
    def __create_labeling_page_df(st):
        data = st.session_state.working_labeling_data

        LabelingPage.__show_header(st, data)

        blocks, all_codes = LabelingPage.__separate_blocks(st, data)

        col1, col2 = st.columns([10, 3])
        txt1 = ""
        # print(blocks)
        for idx, (type, block) in enumerate(blocks):

            for record in block:
                # each block contain list of records {line_num:line_data}
                for line_num, line_data in record.items():
                    # st.write(f"{idx} {type}")
                    txt1 += f"""{line_num}. {line_data["content"]}\n"""

            if type == "comment":
                with col2.popover(f"{list(block[0].items())[0][0]}: Edit annotation"):
                    # Create a list to store dictionaries
                    records = []
                    sufficiency_records = []

                    for record in block:
                        for line_num, line_data in record.items():
                            if "labels" in line_data:
                                # Get the labels from the line_data
                                labels = line_data['labels']
                                # Create a dictionary with line_num and label values

                                if 'sufficient' in labels:
                                    sufficiency_records.append((line_num, labels['sufficient']))
                                else:
                                    record_dict = {'line':line_num}
                                    for crit, val in labels.items():
                                        record_dict[crit] = bool(val)
                                    records.append(record_dict)

                    # Convert the list of dictionaries to a DataFrame
                    df = pd.DataFrame(records)
                    # Render the labels DataFrame using data_editor
                    edited_labels = st.data_editor(df, hide_index=True, key=f'lab_{idx}')

                    # Render the 'sufficient' label dataframe
                    st.write("Annotate for the sufficiency of generated responses")
                    edited_sufficient = st.data_editor(pd.DataFrame(
                        [{'line': l, 'sufficient':bool(val)} for (l, val) in sufficiency_records]),
                        hide_index=True,
                        key=f'suf_{idx}'
                    )

                    # update and save the edited labels in session variable to save the data
                    if st.button("save", key=f'but_{idx}'):
                        criteria = list(edited_labels.columns)
                        criteria.remove('line')
                        for idx, row in edited_labels.iterrows():
                            line_num = row['line']
                            for crit in criteria:
                                crit_val = int(row[crit])
                                st.session_state.working_labeling_data["lines"][line_num]["labels"][crit] = crit_val

                        # Update the sufficient label
                        for idx, row in edited_sufficient.iterrows():
                            line_num = row['line']
                            val = int(row['sufficient'])
                            st.session_state.working_labeling_data["lines"][line_num]["labels"]['sufficient'] = val

                        LabelingPage.__save_annotation(st, create_new=None)

            else:
                txt = ""
                for record in block:
                    for line_num, line_data in record.items():
                        txt += f"""{line_num}. {line_data["content"]}\n"""
                        # txt += f"""{line_num}\n"""

                # col1.code(txt)
                col2.markdown(f"<br>" * len(block), unsafe_allow_html=True)
                # col2.code(txt)

                txt = ""
        col1.code(txt1)


    @staticmethod
    def __create_labeling_page(st):
        data = st.session_state.working_labeling_data

        LabelingPage.__show_header(st, data)

        blocks, all_codes = LabelingPage.__separate_blocks(st, data)

        col1, col2 = st.columns([10,3])
        txt1 = ""
        # print(blocks)
        for idx, (type, block) in enumerate(blocks):

            for record in block:
                # each block contain list of records {line_num:line_data}
                for line_num, line_data in record.items():
                    # st.write(f"{idx} {type}")
                    txt1 += f"""{line_num}. {line_data["content"]}\n"""

            if type == "comment":

                with col2.popover(f"{list(block[0].items())[0][0]}: Edit annotation"):
                    # st.write(f"{record} hello world")
                    txt = ""
                    for record in block:
                        for line_num, line_data in record.items():
                            if "labels" in line_data:
                                items = line_data['labels'].items()
                                cols = st.columns(len(items))
                                for i, (crit, val) in enumerate(items):
                                        if cols[i].checkbox(crit, value = bool(int(val)), key=f"chk_{line_num}_{i}"):

                                            st.session_state.working_labeling_data["lines"][line_num]["labels"][crit] = 1
                                            # data["lines"][line_num]["labels"][crit] = 1
                                        else:
                                            st.session_state.working_labeling_data["lines"][line_num]["labels"][crit] = 0
                                            # data["lines"][line_num]["labels"][crit] = 0

                            st.write(f"""{line_num}. {line_data["content"]}""")

                            txt += f"""{line_num}. {line_data["content"]}\n"""

                txt = ""
            else:
                txt = ""
                for record in block:
                    for line_num, line_data in record.items():
                        txt += f"""{line_num}. {line_data["content"]}\n"""
                        # txt += f"""{line_num}\n"""

                # col1.code(txt)
                col2.markdown(f"<br>"*len(block), unsafe_allow_html=True)
                # col2.code(txt)

                txt = ""
        col1.code(txt1)


    @staticmethod
    def load_labeling_page(st):
        if st.session_state.working_labeling_data is not None:
            # LabelingPage.__create_labeling_page(st)
            LabelingPage.__create_labeling_page_df(st)
        else:
            st.write("Chose existing file or upload a new file to label")
