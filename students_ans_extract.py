from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st
from openai import OpenAI
import json
import os
import sys
from dotenv import load_dotenv
from tenacity import retry,stop_after_attempt,wait_random_exponential
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

api_key = os.environ['openai_api_key']
azure_endpoint = os.environ['azure_endpoint']
azure_key = os.environ['azure_key']

class PDFStudentAnswerExtractor:
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
             {"role": "system", "content": ''' You are an expert text formater. You will be provided a text file containing student answers of different sections like MCQS, fill in the blanks and descriptive answers.
              Your task is to format the text clearly so that it is understandable. Don't write your answers or don't remove answers. Just give same answers in clear and understandble way. 
              Student_Answers: All the student answers in a formatted.

            Example 1: Consider the following text: 
              
                1) (iv)
                2)
                (i)
                3) (iv)
                4)
                (i)
                #) liv
                6)
            It should be formatted as:
              1) (iv)
              2) (i)
              3) (iv)
              4) (i)
              5) (iv)
              6) student not provided answer.

        Example 2: Consider the following answer text:
              24)
                from the heart
                (a) If we remove the izion are from the
                magnet and then lose it the circuit,
                the electromagnet will still work but
                it will have the less magnetic power.
                (b) If we place a permanent magnet instead
                of an electromagnet, the hammer will
                permanently stick to the it & the bell will
                not ring.
      It should be formatted as:
              24) (a) If we remove the iron core from the magnet and then close the circuit, the electromagnet will still work but it will have less magnetic power. (b) If we place a permanent magnet instead of an electromagnet, the hammer will permanently stick to it & the bell will not ring.
              
        And in the similar way format other questions.      
             '''},
            {"role": "user", "content": f"Format the given text: {data} and give complete formatted answers in response."}
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
                                        "Student_Answers": {
                                            "type": "string",
                                             "Pattern": "Question_Number:\\d+\\ .+ Teacher_Answer$"
                                        },
                                         
                                    },
                                    "required": ["Student_Answers"]
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
            model="gpt-4",
            messages=messages,
            tools=tools,
            seed=100,
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
        try:

            # Use BytesIO object directly
    #     pdf_bytes = uploaded_pdf.read()
            pdf_bytes = uploaded_pdf
            poller = self.document_analysis_client.begin_analyze_document("prebuilt-idDocument", pdf_bytes,)
            result = poller.result()

            # Extract the text content
            content = result.content

            formatted_text = ""
            for page in result.pages:
                for line in page.lines:
                    text = line.content.strip()
                    if text.endswith('.'):
                        formatted_text += f"{text} "
                    else:
                        formatted_text += f"\t{text} "

        # Save extracted text to a file (optional)
            with open("extracted_text3.txt", "w",encoding='utf-8') as text_file:
                text_file.write(content)

            # Analyze answers with GPT model
            st.write(" Students Answers Formating is started 🧑‍💻. Please wait...")
            response = self.analyze_answers("question paper", content)
            print("response of questions extraction model-------->",response)
            fb = json.loads(response)
            all_the_questions = [item['Student_Answers'] for item in fb['feedback']]
           # st.write(all_the_questions)

            file_path = 'Students_answers.txt'
            self.save_questions_to_file(all_the_questions, file_path)
            print(f"Questions saved to {file_path}")
            print("all the answers---------->",all_the_questions)
            st.write("Students Answers Formating is done ✅.")
            return all_the_questions
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Main Streamlit app
def main():
    st.title("PDF Text Extractor")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if st.button("Extract"):
        if uploaded_pdf is not None:
            extractor = PDFStudentAnswerExtractor(
                openai_api_key= api_key,
                azure_endpoint=azure_endpoint,
                azure_key= azure_key
            )
            listofanswers = extractor.analyze_read(uploaded_pdf)
            print(listofanswers)
            st.write(listofanswers)

if __name__ == "__main__":
    main()

