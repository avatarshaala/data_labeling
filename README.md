### Data Labeling Application

An application designed for annotating data generated by the Programming Tutor Feedback Generator.

### Data Annotation Process

1. **Login**: Access the application using your username and password.

2. **File Upload**:
   - If LLM-generated Java files haven't been converted to JSON, simply browse and upload the file for annotation.
   - Upon upload, the file is automatically saved in JSON format, prefixed with the user and date (user.date.filename.json).

3. **Annotation Interface**:
   - Once the file is uploaded, the annotator interface displays:
     1. AI-generated scaffolding along with the program code.
     2. JSON representation of the program with default annotations.
     3. Section-wise "Edit annotation" button.

4. **Editing Annotations**:
   - Clicking the "Edit annotation" button allows for the population of line numbers, annotation criteria, and labels for those criteria.
   - Evaluation criteria include:
     1. **Relevancy**: Determines if the scaffolding is pertinent to the code block and enhances learning.
     2. **Quality**: Evaluates the grammatical correctness and linguistic aspects of the scaffolding.
     3. **Sufficiency**: Assesses whether the number of scaffolding instances is adequate for understanding the code or if more are needed to assist students with potential learning difficulties.

5. **Saving Annotations**: After editing the annotation, ensure to save the changes.

6. **Completion**:
   - Once the annotation process is complete, we will utilize a separate data analyzer application (already developed) to compute and visualize various metrics.


---

This application streamlines the process of annotating data, facilitating further analysis and insights to enhance programming education.
___________________