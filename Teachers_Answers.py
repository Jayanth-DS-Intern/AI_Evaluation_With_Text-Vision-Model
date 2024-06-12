from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st
from openai import OpenAI
import json
import os
import sys,time
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_random_exponential
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

api_key = os.environ['openai_api_key']
azure_endpoint = os.environ['azure_endpoint']
azure_key = os.environ['azure_key']

class PDFTeacherAnswerExtractor:
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

    def extract_questions(self, response_content):
        sections = {}
        current_section = None

        for line in response_content.split('\n\n'):
            if line.startswith('SECTION-'):
                current_section = line.strip()
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        extracted_questions = []
        for section, questions in sections.items():
            for question in questions:
                extracted_questions.append(question)

        return extracted_questions

    def evaluation(self, feedback, topic):
        return json.dumps({"feedback": feedback, "topic": topic})

    def analyze_answers(self, topic, data):
        messages = [
             {"role": "system", "content": '''  You will be provided a solutions paper. It contains different sections like MCQS, Fill in the blank questions, 2 marks, 3 Marks questions and case based questions with there answers.
            Follow the guidelines:
              
           >For questions with multiple parts (e.g., 17.a, 17.b, ... 17.h), extract all answers together at the same time under the main Question Number. 
            - Combine the answers for the sub-parts into a single string, separated by commas.
            - For example, if question 17 has parts a, b, c, ..., h, and the answers are "a. Convex", "b.sewers", "c.biotic", ..., "h.blue", then extract them as:
                "Teacher_Answer": "a.Convex, b.sewers, c.biotic, ..., h.blue"
                "Question_Number": 17
              
           > Some questions with two parts (eg. 33.a or 33.i, 33.b or 33.ii), extract all answers together at the same time with the main quetsion number like above example
           - For example, if question  33 has parts a,b and answers are "a. No, I don’t think Neeta brought about the pollination of the marigold flowers.....", "b.Arteries carry blood away from the heart to various organs in the body....", respectively then extract as:
               "Teacher_Answer": "a. No, I don’t think Neeta brought about the pollination of the marigold flowers.....,b.Arteries carry blood away from the heart to various organs in the body...."
              
           > Ignore any explanations or additional information.
           > Teacher_Answer: The answer given in the solutions paper.
           > Question Number: The Number of the question for the answer is Teacher_Answer.
           
    
             '''},
            {"role": "user", "content": f"Your job is to extract the only answers with question number from all the sections of the given solutions paper:{data}. Do not provide none as response. You must call tools"}
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
                                        "Teacher_Answer": {
                                            "type": "string",
                                             "Pattern": "Question_Number:\\d+\\ .+ Teacher_Answer$"
                                        },
                                        "Question_Number":{
                                            "type":"number"
                                        }
                                         
                                    },
                                    "required": ["Teacher_Answer","Question_Number"]
                                }
                            }
                        },
                        "required": ["topic", "feedback"]
                    }
                }
            }
        ]

         
        
        # @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
        # def exponential_backoff():
        #     print("Sending request to OpenAI API")
        #     response = self.openai_client.chat.completions.create(
        #     model="gpt-4-1106-preview",
        #     messages=messages,
        #     tools=tools,
        #     seed=100,
        #     top_p=0.0000000000000000000001,
        #     temperature=0,
        #     tool_choice="required"
        #         )
        
        #     response.raise_for_status()  # Ensure we raise an error for bad responses
        #     print("Received response from OpenAI API")
        #     return response
        # response = exponential_backoff()

        response = self.openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=tools,
            seed=10,
            top_p=0,
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
        # Use BytesIO object directly
        s_t1 = time.time()
        pdf_bytes = uploaded_pdf.read()

        poller = self.document_analysis_client.begin_analyze_document("prebuilt-read", pdf_bytes)
        result = poller.result()

        # Extract the text content
        content = result.content
        e_t1 = time.time()
        print(f"Time taken to extract the teachers text: {e_t1-s_t1}")
       # Save extracted text to a file (optional)
        # with open("extracted_text2.txt", "w",encoding='utf-8') as text_file:
        #     text_file.write(content)

        # Analyze answers with GPT model
        st.write("Teachers Answers Extraction is started 🧑‍💻. Please wait...")
        s_t2 = time.time()
        response = self.analyze_answers("question paper", content)
        e_t2 = time.time()
        print(f"Time taken to extract list of teachers answers:{e_t2-s_t2}")
        print("response of questions extraction model-------->",response)
        fb = json.loads(response)
        print("fb---------->",fb)
        all_the_answers = [item for item in fb['feedback']]
        # st.write(all_the_questions)

        file_path = 'listofanswers.txt'
        self.save_questions_to_file(all_the_answers, file_path)
        print(f"Questions saved to {file_path}")
        print("all the answers---------->",all_the_answers)
        st.write("Teachers Answers Extraction is done ✅.")
        return all_the_answers

# Main Streamlit app
def main():
    st.title("PDF Text Extractor")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if st.button("Extract"):
        if uploaded_pdf is not None:
            extractor = PDFTeacherAnswerExtractor(
                openai_api_key= api_key,
                azure_endpoint=azure_endpoint,
                azure_key= azure_key
            )
            listofanswers = extractor.analyze_read(uploaded_pdf)
            print(listofanswers)
            st.write(listofanswers)

if __name__ == "__main__":
    main()

