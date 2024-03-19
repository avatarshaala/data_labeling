from st_click_detector import click_detector
import streamlit as st

class ClickableGrid(object):

    def __init__(self, key, labels, label_key="labels"):

        '''
        besides content and type, every record has labels:{} as actual label of record, label_key="labels" is provided
        labels is in following format
        "12": {
            "content": "",
            "type": "",
            "labels": {"sufficient": 0}
        },
        "13": {
            "content": "",
            "type":"",
            "labels": {"relevant": 0, "quality": 1}
        },
        '''

        self.key = key
        self.__labels = {}
        #initialize labels
        for item, specs in labels.items():
            self.__labels[item] = {}
            for spec, value in specs[label_key].items():
                self.__labels[item][spec] = value

        self.__clickable_cells = ""

        for item, specs in self.__labels.items():
            self.__clickable_cells += f"{item}. "
            for spec, value in specs.items():
                id = f"{self.key}_{item}_{spec}"
                self.__clickable_cells += f"""<a href='' id={id} target='_top'>{spec}?</a>&nbsp;&nbsp;&nbsp;"""

            self.__clickable_cells += "<br>"

            self.__click_detector = click_detector

        # self.update_annotations(labels)

    def grid_click_detector(self):

        clicked = self.__click_detector(self.__clickable_cells, key=f"{self.key}")
        itm, spc = None, None
        if clicked:
            print("__________________________")
            itm, spc = clicked.split("_")[-2:]
            # print(clicked, (itm, spc))
            # toggle the annotation value
            self.__labels[itm][spc] = 1 - self.__labels[itm][spc]
            print("ITM SPC: ",itm, spc)
        return (itm,spc), self.__labels

    def annotation_display(self):
        self.__display = ""
        for item, specs in self.__labels.items():
            self.__display += f"{item}.&nbsp;&nbsp;&nbsp;&nbsp;"
            for spec, value in specs.items():
                self.__display += f"{value}&nbsp;&nbsp;&nbsp;&nbsp;"

            self.__display += "<br>"

        return self.__display, self.__labels

    @staticmethod
    def clickable_grid(st, key, labels, label_key="labels"):
        if key not in st.session_state:
            st.session_state.key = ClickableGrid(key, labels, label_key)

        return st.session_state.key


# col1, col2 = st.columns([4, 10])
#
#
# labels = {
#     "12": {"labels": {"sufficient": 0}},
#     "13": {"labels": {"relevant": 0,"quality": 1 }},
#     "14": {"labels": {"relevant": 0,"quality": 1}}
# }
#
# key = "cg_1"
# # if key not in st.session_state:
# #     st.session_state.key = ClickableGrid(key, labels, "labels")
#
# # cg = st.session_state.key
# cg = ClickableGrid.clickable_grid(st, key, labels, "labels")
# with col1:
#     clicked, labs = cg.grid_click_detector()
#
# with col2:
#     display, labs = cg.annotation_display()
#     st.markdown(display, unsafe_allow_html=True)
#
#
# key = "cg_4"
#
# cg1 = ClickableGrid.clickable_grid(st, key, labels, "labels")
#
# clicked, labels = cg1.grid_click_detector()
# st.write(clicked, labels)