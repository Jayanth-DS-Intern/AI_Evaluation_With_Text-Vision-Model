from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st
from openai import OpenAI
import json
import os,time
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

api_key = os.environ['openai_api_key']
azure_endpoint = os.environ['azure_endpoint']
azure_key = os.environ['azure_key']

class PDFQuestionExtractor:
    def __init__(self, openai_api_key, azure_endpoint, azure_key):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=azure_endpoint, credential=AzureKeyCredential(azure_key)
        )
        self.chatgpt_url = "https://api.openai.com/v1/chat/completions"
        self.chatgpt_headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

    def save_questions_to_file(self, questions, file_path):
        with open(file_path, "w") as file:
            for question in questions:
                file.write(f"{question}\n")


    def evaluation(self, feedback, topic):
        return json.dumps({"feedback": feedback, "topic": topic})

    def analyze_answers(self, topic, data):
        messages = [
             {"role": "system", "content": ''' You're a expert text extractor you will be provided a text file your task is to extract all the questions with question numbers by Considering the following guidelines while extracting questions:

When you are extrating the questions you should also extract the maximum number of marks for that question, For a particular section no.of questions and Marks for each will given at the starting of the section, all the questions below that section will have carry same marks(remember).      

Your job Also to remember that carefully see complete question and any question says like "draw","sketch","diagram" then only you have to label that question as diagram need: "Yes" otherwise "No".

Remember, Rememeber and Remember You must give follow:
Each question must follow this format: 'Question_Number . Question text (MaximumMarks:Number, Diagram_Needed:Yes/No)'.
response example: "1. Define photosynthesis? (MaximumMarks:2, Diagram_Needed:No)"

👉 Remember all questions strictly follow the specified format.
                   
And if the question already given graph or diagram then don't label that question as Diagram_Needed as yes.
There will be different formats of questions (e.g., MCQs for 1 mark, fill in the blanks for 8 marks as they contain 8 parts, case-based questions, choices questions), each with its own marking system.

You have to extract questions from all these sections:
              
1. Compound Questions: If a question contains two or more parts, extract all of them as a single question.

Example: "11. What is orested Principle? Draw the diagram of his experiment ?" should be extracted as:
"11. What is orested Principle? Draw the diagram of his experiment? ('MaximumMarks': 2, 'Diagram_Needed':Yes)"
         
2. Contextual Questions with Human Characters: Include any human characters or contextual information provided within the question, as they are essential for framing the complete question.

Example: "12. jayanth pulled a plant from the ground and placed it in a pot of soil. However, most of the root hairs were present in the ground and the plant did not grow. Justify your answer with a suitable reason. " should be extracted as:
"12. jayanth pulled a plant from the ground and placed it in a pot of soil. However, most of the root hairs were present in the ground and the plant did not grow. Justify your answer with a suitable reason.('MaximumMarks': 2, 'Diagram_Needed':No)"

And If you find text like this than extract the following question formats in the following way: Answer the following questions.(6 x 4 - 24 marks) . It means all the 6 question below this heading will carry 4 marks.
             
3. Embedded Questions: Extract questions even if they are embedded within a statement or scenario.

Example: "15.While conducting an experiment diagram given, Priya observed some changes. What did she observe and why did those changes occur?" should be extracted as:
""15.While conducting an experiment diagram given, Priya observed some changes. What did she observe and why did those changes occur?('MaximumMarks': 4, 'Diagram_Needed':No)"
         
4.List-Based Questions: If the question includes multiple parts or choices, extract them all together.

Example: "14. i or a)What are the different types of renewable energy sources? 
         ii or b)Explain each type briefly." should be extracted as:
"14. i or a)What are the different types of renewable energy sources? ii or b) Explain each type briefly.('MaximumMarks': 4, 'Diagram_Needed':No)"
         

5. Narrative-Based questions with Multiple Parts: Include all parts of the question, especially if they involve detailed scenarios and multiple sub-questions.

Example: "31. Rob started running towards the north, starting from his home. He covered 20 metres in 2 minutes. Next, he increased his speed and ran back 20 metres in 1 minute. Then, he turned north again and ran for 2 minutes with the starting speed, covering 20 metres in this direction.
a) Draw a distance-time graph for Rob.
b) What is the average velocity of Rob?" should be extracted as:
"31. Rob started running towards the north, starting from his home. He covered 20 metres in 2 minutes. Next, he increased his speed and ran back 20 metres in 1 minute. Then, he turned north again and ran for 2 minutes with the starting speed, covering 20 metres in this direction.a) Draw a distance-time graph for Rob.b) What is the average velocity of Rob?. ('MaximumMarks': 4, 'Diagram_Needed':Yes)"

6. Choice questions**: Extract both the questions at a time.
 - Example: 16. Example: "India is fortunate to have fairly rich and varied mineral resources but there are unevenly distributed". Analise the statement
                    [ OR ]
            "Energy saved is the energy produced". Elaborate the statement with examples.
        Then It should be extracted as:Example:16. "India is fortunate to have fairly rich and varied mineral resources but there are unevenly distributed". Analise the statement [or] "Energy saved is the energy produced". Elaborate the statement with examples. ('MaximumMarks': 5, 'Diagram_Needed':No)"
              
7. Questions with bit numbers or section number: Iv) Answer the following questions (1 X 5= 5) It means marks should be divided between two questions since for this complete bit they have given 5 marks so the division of marks is in your hands, based on the difficulty level you assign, It doesn't mean that both questions will carry 5 marks
    similarly for other bits like III) Answer the following questions 6*4=24 It means in this bit there are 6 questions and each question carry 4 marks. Do provide the bit number as well with question number.
              
 8. Case Based Questions: You have to extract all the questions that are included in the case.
    For example: The reflecting surface of a spherical mirror may be curved inwards or outwards. A spherical mirror whose reflecting surface is curved inwards, that is faced towards the centre of the sphere is called a concave mirror. A spherical mirror whose reflecting surface is curved outward is called convex mirror.
            i. A mirror with a surface then curves inward like the inside of a bowl :-
              --options--
            ii. Convex mirror produce :-
            --options--
            iii. Convex mirrors are :-
             --options--
    Now you have to extract all the question i,ii and iii at a time.
              
9. Multiple choice questions(MCQS):
    For example: 4. How can the strength of the current in a coil be increased?
    (i) By using multiple cells connected to the coil
    (ii) By reducing the number of turns in the coil
    (iii) By using non-magnetic materials for the core
    (iv) By decreasing the strength of the current
              
    you have to extract as :  "4. How can the strength of the current in a coil be increased? (i) By using multiple cells connected to the coil (ii) By reducing the number of turns in the coil (iii) By using non-magnetic materials for the core (iv) By decreasing the strength of the current  (MaximumMarks:1, Diagram_Needed:No)"
              
10. Fill in the blanks:
    For example if you saw text like this: Fill in the blanks.
                                          .....
                                            17) a) mirrors are commonly used as rear-view mirrors in vehicles.
                                            (2 x 8 = 16 marks )
                                            Answer: Convex Solution: (Convex mirrors are commonly used as rear-view (wing) mirrors in vehicles.) Give 1 mark for the correct answer.
                                            b) The wastewater from housing communities or public places is taken away by a network of pipes called .
                                            Answer:
                                            .... so on
              It means both questions 17) a),b),...h)('MaximumMarks':8, 'Diagram_Needed':Yes) and 18) a),b) ...h) ('MaximumMarks':8, 'Diagram_Needed':Yes) carry 8 marks since in 17 and 18 eight subquestions are present.
              
Question Number: For example IF the given text is like this:
                 25 VERY SHORT ANSWER QUESTIONS [4 x 2 = 8m] SECTION - B
                The arrangement made by Belgian leaders is very innovative. Give any two elements of Belgian model.
                2
                22
                Explain how the global transfer of disease in the pre - modern world helped in the colonization of Americans
        Then question number for this question The arrangement made by Belgian leaders is very innovative. Give any two elements of Belgian model. is 21 and
             for this question: Explain how the global transfer of disease in the pre - modern world helped in the colonization of Americans is 22.
              
                         
Note1: The values for MaximumMarks and Diagram_Needed should not be strings they should be like this: ('MaximumMarks':2, 'Diagram_Needed':No) not like this: ('MaximumMarks': '2', 'Diagram_Needed': 'No')
            
Note2: Provide the same questions numbers and MaximumMarks that are present in given data:{data}'''},
            {"role": "user", "content": f"Your main job is to extract all the questions from the provided data: {data}. Do not provide none as response. You must call tools and do provide the system_fingerprint value."}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "evaluation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "feedback": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {

                                        "MaximumMarks": {"type": "number"},
                                        "Diagram_Needed": {"type": "string"},
                                        "Question_Number": {"type": "number"},

                                        "question": {
                                            "type": "string",
                                            "pattern": "\\d+\\ .+ \\(MaximumMarks:\\d+, Diagram_Needed:(Yes|No)\\)$"
                                        },
                                         
                                    },
                                    "required": ["question", "Question_Number", "MaximumMarks", "Diagram_Needed"]
                                }
                            }
                        },
                        "required": ["topic", "feedback"]
                    }
                }
            }
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=tools,
            seed=92,
            top_p=0.0000000000000000000001,
            temperature=0,
            tool_choice="required"
        )
        
         
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {"evaluation": self.evaluation}
            function_name = tool_calls[0].function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_calls[0].function.arguments)
            function_response = function_to_call(
                feedback=function_args.get("feedback"),
                topic=function_args.get("topic"),
            )
            return function_response

    def analyze_read(self, uploaded_pdf):
        try:

            # Use BytesIO object directly
            s_t = time.time()
            pdf_bytes = uploaded_pdf.read()

            poller = self.document_analysis_client.begin_analyze_document("prebuilt-read", pdf_bytes)
            result = poller.result()

            # Extract the text content
            content = result.content
            e_t = time.time()
            print(f"Time taken to extract questions text using from recognizer:{e_t-s_t} sec")
        # Save extracted text to a file (optional)
            with open("extracted_text2.txt", "w",encoding='utf-8') as text_file:
                text_file.write(content)

            # Analyze answers with GPT model
            st.write("Questions Extraction is started 🧑‍💻. Please wait...")
            start_time = time.time()
            response = self.analyze_answers("question paper", content)
            end_time = time.time()
            print(f"Time taken to get the list of questions:{end_time-start_time} sec")
        #  print("response of questions extraction model-------->",response)
            fb = json.loads(response)
            all_the_questions = [item['question'] for item in fb['feedback']]
            print(all_the_questions)
            file_path = 'list_of_questions_for_vision.txt'
            self.save_questions_to_file(all_the_questions, file_path)
            print(f"Questions saved to {file_path}")
            print("all the questions---------->",all_the_questions)
            st.write("Questions Extraction is done ✅.")
            return all_the_questions
        except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Main Streamlit app
def main():
    st.title("PDF Text Extractor")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if st.button("Extract"):
        if uploaded_pdf is not None:
            extractor = PDFQuestionExtractor(
                openai_api_key= api_key,
                azure_endpoint=azure_endpoint,
                azure_key= azure_key
            )
            listofquestions = extractor.analyze_read(uploaded_pdf)
            print(listofquestions)
            st.write(listofquestions)

if __name__ == "__main__":
    main()

