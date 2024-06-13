import streamlit as st
import openai
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import OpenAI
import json,tempfile,ast
import io,time,base64,requests,os,fitz
from extract_ques import PDFQuestionExtractor
from students_ans_extract import PDFStudentAnswerExtractor
from Teachers_Answers import PDFTeacherAnswerExtractor
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_random_exponential,
)  # for exponential backoff
from logger import logger
load_dotenv()
import concurrent.futures

# Set up OpenAI API key
# api_key = os.environ['openai_api_key']
# azure_endpoint = os.environ['azure_endpoint']
# azure_key = os.environ['azure_key']

api_key = st.secrets['openai_api_key']
azure_endpoint = st.secrets['azure_endpoint']
azure_key = st.secrets['azure_key']

client = OpenAI(
  api_key= api_key,  # this is also the default, it can be omitted
)

    
tools1 = [
        {
            "type": "function",
            "function": {
            "name": "evaluation",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string"
                    },
                    "Question_Number": {
                         "type": "number"
                    },
                    "feedback": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string"
                                },
                     "Teacher_Answer": {
                                    "type": "string"
                                },
                    "Student_Answer": {
                                    "type": "string"
                                },
				"Accuracy": {
                                    "type": "string"
                                },
				"Relevance": {
                                    "type": "string"
                                },
				"Completeness": {
                                    "type": "string"
                                },
				"Depth of Understanding": {
                                    "type": "string"
                                },
				"Clarity of Expression": {
                                    "type": "string"
                                },
				"Use of Examples": {
                                    "type": "string"
                                },
				# "Grammar and Mechanics": {
                #                     "type": "string"
                #                 },
				"Overall Quality": {
                                    "type": "string"
                                },
				"No of words used": {
                                    "type": "number"
                                },
                      "feedback" : {
                                    "type":"string"
                                },
                                "Overal Score": {
                                    "type": "number",
                 
                                },                            },
                            "required": ["question", "answer","Accuracy","Relevance","Completeness","Depth of Understanding","Clarity of Expression","Use of Examples","Overall Quality","feedback","No of words used","Overal Score"]
                        }
                    }
                },
                "required": ["topic", "Question_Number","feedback"]
            }
        }
        }
    ]

tools2 = [
        {
            "type": "function",
            "function": {
            "name": "evaluation",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string"
                    },
                    "Question_Number": {
                         "type": "number"
                    },
                    "feedback": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string"
                                },
                     "Teacher_Answer": {
                                    "type": "string"
                                },
                    "Student_Answer": {
                                    "type": "string"
                                },
				"Accuracy": {
                                    "type": "string"
                                },
				"Relevance": {
                                    "type": "string"
                                },
				"Completeness": {
                                    "type": "string"
                                },
				"Depth of Understanding": {
                                    "type": "string"
                                },
				"Clarity of Expression": {
                                    "type": "string"
                                },
				"Use of Examples": {
                                    "type": "string"
                                },
				"Overall Quality": {
                                    "type": "string"
                                },
				"No of words used": {
                                    "type": "number"
                                },
                      "feedback" : {
                                    "type":"string"
                                },
                        "Diagram":{
                                    "type":"string"
                        },
                                "Overal Score": {
                                    "type": "number",
                 
                                },                            },
                            "required": ["question", "answer","Accuracy","Relevance","Completeness","Depth of Understanding","Clarity of Expression","Use of Examples","Grammar and Mechanics","Diagram","Overall Quality","feedback","No of words used","Overal Score"]
                        }
                    }
                },
                "required": ["topic","Question_Number","feedback"]
            }
        }
        }
    ]
    

class VisionModelAnalyzer:
    def __init__(self, headers):
        self.headers = headers
        self.required_image_paths = []
        self.current_index = 0

    def extract_images_from_pdf(self,pdf_path, output_folder):
        # Create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # List to store paths of extracted images
        image_paths = []

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through each page of the PDF
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)

            # Get the images on the page
            image_list = page.get_images(full=True)

            # Extract and save each image
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_extension = base_image["ext"]
                image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"

                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                
                # Append the image path to the list
                image_paths.append(image_filename)
        
        # Return the list of image paths
        return image_paths

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string

    def filter_images_by_page_numbers(self,page_numbers,image_paths):
        filtered_paths = []
          
        for page in page_numbers:
            for path in image_paths:
                if page in path:
                    filtered_paths.append(path)
        return filtered_paths
                     
    
    def evaluation(self, feedback, topic):
        return json.dumps({"feedback": feedback, "topic": topic})
    
    output_images_folder = tempfile.mkdtemp(prefix="output_images_folder")

     
              
class TextModelAnalyzer:
    def __init__(self):
          
          pass
    
    def evaluation(self,feedback,topic):
        return json.dumps({"feedback": feedback, "topic":topic})
    
            
    def main(self,questions_pdf, teachers_pdf, answers_pdf):
         pass
    

def extract_images_from_pdf(pdf_path, output_folder):
        # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List to store paths of extracted images
    image_paths = []

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page of the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Get the images on the page
        image_list = page.get_images(full=True)

        # Extract and save each image
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_extension = base_image["ext"]
            image_filename = f"{output_folder}/page{page_number+1}.{image_extension}"

            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            
            # Append the image path to the list
            image_paths.append(image_filename)
    
    # Return the list of image paths
    return image_paths

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string
         
def analyze_answers_vision_model(question,temp_pdf_path, teacher_answer,temp_dir):
         

        # Extract images from the PDF
        image_paths = extract_images_from_pdf(temp_pdf_path, output_folder=temp_dir)
        print("all image paths---->",image_paths)

        
        # Filter images based on PAGE_NUMBERS
        # self.required_image_paths = self.filter_images_by_page_numbers(PAGE_NUMBERS,self.image_paths)
        # print("required_image_paths",self.required_image_paths)


        # Encode images to base64
        base64_images = [encode_image(image_path) for image_path in image_paths]
         

        # image_path = self.required_image_paths[self.current_index]
        # self.current_index += 1

      #  base64_image = self.encode_image(image_path)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": """You are going to evaluate a student's handwritten answers against teacher answer based on a provided question. Follow these steps to accurately complete the evaluation and provide constructive feedback:
             
    Carefully read through the Question to understand what is being asked.
    Analyze each of the Student_Answers in image scene, comparing them to the corresponding Teachers Answer .
    If the question requires the diagram then carefully look at the student answers in answers scene and evaluate.
    > Carefully check the students answers scene for graph or diagram needed.
                 
 👉 (Remember): after taking the question number from the given question do not remove it. Give the complete question given to you.    
                                 
    As a teacher Consider the following guidelines while consdering and evaluating the given question:

   1. Compound Questions: If a question contains two or more parts, extract all of them as a single question.

    Example: "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)" should be extracted as:
    "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)"

    2. Multiple questions in a single question:
        example:"31. Rob started running towards the north, starting from his home. He covered 20 metres in 2 minutes. Next, he increased his speed and ran back 20 metres in 1 minute. Then, he turned north again and ran for 2 minutes with the starting speed, covering 20 metres in this direction. a) Draw a distance-time graph for Rob. b) What is the average velocity of Rob? ('MaximumMarks':4, 'Diagram_Needed':Yes)
                 When questions like above are given you have to clearly serach for the question in it that needs diagram and you have to evaluate other parts with answer provided. 
        remember: you must give response.
                 
    You should give the complete question with Question Number, Diagram_Needed and MaximumMarks tag also Don't forget it.
         
    Give the output that should contain Question,overall score, and following metrics:

    1)Accuracy: Determine the correctness of the information provided in the response. Are the facts and concepts presented accurate?

    2)Relevance: Assess whether the student's response directly addresses the question asked. Is the information provided relevant to the topic being discussed?

    3)Depth of Understanding: Evaluate the depth of the student's understanding of the topic. Does the response demonstrate a deep understanding of the concepts, theories, and principles involved?

    4)Completeness: Consider whether the response covers all aspects of the question comprehensively. Are all relevant points addressed, or are there significant omissions?

    5)Clarity of Expression: Evaluate the clarity and coherence of the student's writing. Is the response well-organized and easy to follow? Are ideas expressed clearly and concisely?

    6)Use of Examples: Assess whether the student provides relevant examples to support their arguments or explanations. Do the examples enhance understanding and illustrate key points effectively?

    7)Critical Thinking: Analyze the student's ability to critically evaluate and analyze information. Does the response demonstrate critical thinking skills, such as the ability to analyze, evaluate, and synthesize information?

    8)Originality: Consider whether the student offers original insights or perspectives on the topic. Does the response demonstrate creativity or independent thinking?

    9)Integration of Sources: If applicable, evaluate the student's ability to integrate information from multiple sources (e.g., textbooks, articles, lectures) into their response. Are sources properly cited and integrated into the discussion?

    10)Argumentation and Logic: Assess the strength of the student's arguments and the logical coherence of their reasoning. Are arguments well-supported with evidence and logical reasoning?

    11)Diagram : Provide detailed feedback on student's handwritten diagram, explaining why it is correct or incorrect along with errors or improvements needed.
         
    12)Overall Quality: Provide an overall assessment of the quality of the student's response. Consider all of the above metrics in conjunction to determine the overall effectiveness of the response.

    Make sure to provide all metrics.If not applicable ignore the specific metric by saying N/A. ensuring to include a newline character at the end to properly format your result.Display both Question and Answer written.

    Example response:
   "1. Question: What is the Role of roughages in the Alimentary tract. (MaximumMarks:2, Diagram_Needed:No)?
    Teacher_Answer: The main functions of roughages are: They help in movement of food through the alimentary canal as well as in proper bowel movement, Helps to prevent constipation and to get rid of undigested food, It also helps in retaining water in the body.       
    Student_Answer: Promoting Digestive Health, Aiding in weight management, Regulating Blood Sugar level, lowering cholesterol levels, preventing colon Cancer

    Overall Score: 1.5/2

    1) Accuracy:  High
    2) Relevance:  High
    3) Depth of Understanding:  Satisfactory
    5) Clarity of Expression: High
    6) Use of Examples: N/A
    7) Critical Thinking: N/A
    8) Originality: N/A
    9) Integration of Sources: N/A
    10) Argumentation and Logic: N/A
   
    12) Feedback: Great job! Your answer accurately covers the roles of roughage but could benefit from more detail on how each role is performed.

    13) Overall Quality:  High
         Note: you must use tools always!and for the value of topic you do not take the question. Topic name is the list of concepts and sub concepts from which the question is curated.
                """},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"Find the relevant answer for the question: {question} in the given image and evaluate aganist teacher answer: {teacher_answer['Teacher_Answer']}give a response accurately. You MUST USE TOOLS!. If you don't find the answer and diagram it means student has not written the answer, then move to next question by saying the student has not written the answer. Also remember not to create new or new sentences from the students answers, extract them completely and evaluate it. If the question needs a diagram, do evaluate it and give feedback on the diagram as well. You must always use tools!"
                    },
                    *[
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}","detail": "high"}}
                        for base64_image in base64_images
                    ]
                ]}
            ],
            "max_tokens": 1000,
            "tools": tools2,
        }

        print("Given Question ------>", question)
        

        @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
        def vision_response(headers,payload):
            print("sent request to openai")
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status() 
            return response
        
        response = vision_response(headers=headers,payload=payload)
         
        print("---------------response.json()-------------------")
        print(response.json())
        response_message = response.json()["choices"][0]["message"]
        tool_calls = response_message['tool_calls']
        if tool_calls:
            available_functions = {
                "evaluation": evaluation,
            }
            function_name = tool_calls[0]['function']['name']
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_calls[0]['function']['arguments'])
            function_response = function_to_call(
                feedback=function_args.get("feedback"),
                topic=function_args.get("topic"),
                Question_Number= function_args.get("Question_Number")
            )
            return function_response
        
headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                }
def evaluation(feedback,topic,Question_Number):
        return json.dumps({"feedback": feedback, "topic":topic, "Question_Number":Question_Number})

def analyze_answers_text_model(question, answers,teacher_answer):
        
        messages = [
            {"role": "system", "content": """You are going to evaluate a student's handwritten answers against Teacher answer based on a provided question. Follow these steps to accurately complete the evaluation and provide constructive feedback:
                
        Carefully read through the Question to understand what is being asked!
        Review the Answer_Sheet to understand the correct and expected answers for each question in the Question_Paper.
        Analyze each of the Student_Answers in images scene especially the MSQS are written in roman numbers, evaluate them carefully by comparing them to the corresponding Teachers Answers .
        Remember,Remember If the question requires the diagram then carefully look at the students answers in images scene and evaluate.
   👉 (Remember): after taking the question number from the given question do not remove it. Give the complete question given to you.             

    Remember: Please evaluate the fill-in-the-blank questions as follows:

        1. Each fill-in-the-blank question consists of 8 sub-questions, labeled from a to h.
        2. The student must answer all sub-questions (a to h).
        3. During the evaluation, compare each sub-question answer from the student with the corresponding sub-question answer from the teacher.
        4. If any of the student's answers do not match the teacher's answers, deduct marks according to the maximum marks of the question.
        5. Only award 8 out of 8 marks if all the student's answers match all the teacher's answers exactly.
                
        As a teacher Consider the following guidelines while consdering and evaluating the given question: {question}:

    1. Compound Questions: If a question contains two or more parts, extract all of them as a single question.

        Example: "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)" should be extracted as:
        "19. How is China rose indicator prepared? What colour change should you expect when this indicator is added to the sodium hydroxide solution?. (MaximumMarks:2, Diagram_Needed:No)"
        
    2. Multiple questions in a single question: You have extract all of them
            For example: "18. a) ---question--- b) ----question---- c) ----question--- .... g) ----question---. (MaximumMarks:2, Diagram_Needed:No)" should be extracted as:
            18. a) ---question--- b) ----question---- c) ----question--- .... g) ----question---. (MaximumMarks:8, Diagram_Needed:No)"
                            
        You should give the complete question with Question Number, Diagram_Needed and MaximumMarks tag also Don't forget it.
            
        Give the output that should contain Question,overall score, and following metrics:
        
        1)Accuracy: Determine the correctness of the information provided in the response. Are the facts and concepts presented accurate?
        
        2)Relevance: Assess whether the student's response directly addresses the question asked. Is the information provided relevant to the topic being discussed?
        
        3)Depth of Understanding: Evaluate the depth of the student's understanding of the topic. Does the response demonstrate a deep understanding of the concepts, theories, and principles involved?
        
        4)Completeness: Consider whether the response covers all aspects of the question comprehensively. Are all relevant points addressed, or are there significant omissions?
        
        5)Clarity of Expression: Evaluate the clarity and coherence of the student's writing. Is the response well-organized and easy to follow? Are ideas expressed clearly and concisely?
        
        6)Use of Examples: Assess whether the student provides relevant examples to support their arguments or explanations. Do the examples enhance understanding and illustrate key points effectively?
        
        7)Critical Thinking: Analyze the student's ability to critically evaluate and analyze information. Does the response demonstrate critical thinking skills, such as the ability to analyze, evaluate, and synthesize information?
        
        8)Originality: Consider whether the student offers original insights or perspectives on the topic. Does the response demonstrate creativity or independent thinking?
        
        9)Integration of Sources: If applicable, evaluate the student's ability to integrate information from multiple sources (e.g., textbooks, articles, lectures) into their response. Are sources properly cited and integrated into the discussion?
        
        10)Argumentation and Logic: Assess the strength of the student's arguments and the logical coherence of their reasoning. Are arguments well-supported with evidence and logical reasoning?
        
        11)Overall Quality: Provide an overall assessment of the quality of the student's response. Consider all of the above metrics in conjunction to determine the overall effectiveness of the response.
        
        12)Overall Feedback: Provide detailed feedback on student's answer, explaning why it is correct or incorrect.
            
        Make sure to provide all metrics.If not applicable ignore the specific metric by saying N/A. ensuring to include a newline character at the end to properly format your result.Display both Question and Answer written.
        
        Example response:
        "1. Question: What is the Role of roughages in the Alimentary tract. (MaximumMarks:2, Diagram_Needed:No)?
        Teacher_Answer: The main functions of roughages are: They help in movement of food through the alimentary canal as well as in proper bowel movement, Helps to prevent constipation and to get rid of undigested food, It also helps in retaining water in the body.       
        Student_Answer: Promoting Digestive Health, Aiding in weight management, Regulating Blood Sugar level, lowering cholesterol levels, preventing colon Cancer

        Overall Score: 1.5/2
        
        1) Accuracy:  High
        2) Relevance:  High
        3) Depth of Understanding:  Satisfactory
        5) Clarity of Expression: High
        6) Use of Examples: N/A
        7) Critical Thinking: N/A
        8) Originality: N/A
        9) Integration of Sources: N/A
        10) Argumentation and Logic: N/A
        12) Feedback: Great job! Your answer accurately covers the roles of roughage but could benefit from more detail on how each role is performed.
    
        13) Overall Quality:  High 

    👉 When you are evauating Mcqs pick the respective MCQ from the given answers do not take all of them.  Topic name is the list of concepts and sub concepts from which the question is curated.
        """},
        
        {"role": "user", "content":f"Now your only task is to evalute the given question:{question} by considering the student relavent answer from answers:{answers} againist the teacher answer: {teacher_answer['Teacher_Answer']} carefully especially in the case of MCQS only pick the respective answer from the choose the correct option section for the given question. Remember Dont give NONE  as the reponse if you don't find the answer in {answers}, go to next question. Marks allotment should be in 0,0.5,1,1.5,2,2.5,3, and so on but not 0.2,1.3,1.7,2.6. You must always use tools."}
        ]

        # @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
        # def exponential_backoff():
        #     print("Sending request to OpenAI API")
        #     response = client.chat.completions.create(
        #     model="gpt-4-1106-preview",
        #     messages=messages,
        #     tools=tools1,
        #     seed=100,
        #     top_p=0.0000000000000000000001,
        #     temperature=0,
        #     tool_choice="required"
        #         )
            
        #     response.raise_for_status()  # Ensure we raise an error for bad responses
        #     print("Received response from OpenAI API")
        #     return response
        # response = exponential_backoff()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools1,
            seed=92,
            top_p=.0000000000000000000001,
            max_tokens=1500,
            temperature=0,
            tool_choice="required"
        )
        print("response-------------->",response)
        response_message = response.choices[0].message
        # response_content = response_message.content
        # return response_content
        #print("response------------",response)
        sf = response.system_fingerprint
    # print("System_fingerprint------------>",sf)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "evaluation": evaluation,
            }  # only one function in this example, but you can have multiple
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            #print("tool_calls-----------------",tool_calls)
            if(1):
                function_name = tool_calls[0].function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_calls[0].function.arguments)
                function_response = function_to_call(
                    feedback=function_args.get("feedback"),
                    topic=function_args.get("topic"),
                    Question_Number=function_args.get("Question_Number")
                )
                return function_response

def chunkify(lst, n):
        """Divide list into chunks of size n."""
        return [lst[i:i + n] for i in range(0, len(lst), n,)]       

def save_questions_to_file(questions, file_path):
        with open(file_path, "w", encoding='utf-8') as file:
            for question in questions:
                file.write(f"{str(question)}\n")

def sort_questions(questions):
     try:    
        # questions = questions.split('\n')
        # question_dicts = [ast.literal_eval(question) for question in questions]
        question_dicts = questions
        sorted_questions = sorted(question_dicts,key=lambda x:x['Question_Number'])
        for question in sorted_questions:
            if "Question_Number" in question:
                del question['Question_Number']
        logger.info("Sucessfully sorted the questions")
     except Exception as e:
          st.error(f"Error Occured in Sort function:{e}")
          logger.exception(f"Error Occured in Sort function:{e}")
          print(e)
     return sorted_questions
     
def convert_string_to_dict(list_of_strings):
    try:
        # Convert each string in the list to a dictionary
        list_of_dicts = [ast.literal_eval(item) for item in list_of_strings]
        logger.info("Successfully Convert the string to dict")
    except Exception as e:
        st.write(e)
        logger.exception(f"Error Occurred in conversion from string to dict: {e}")
        list_of_dicts = []  # Return an empty list in case of an error
    return list_of_dicts

def load_textfile(path):
     with open(path,'r') as file:
                      
                      content = file.read()
     return content

def process_question(analyze_text_function, analyze_vision_function, question, teacher_answer, student_answers, temp_pdf_path, temp_dir):
    if "Diagram_Needed:No" in question:
        response = analyze_text_function( question=question, answers=student_answers, teacher_answer = teacher_answer)
    else:
        response = analyze_vision_function(question = question, temp_pdf_path=temp_pdf_path, teacher_answer = teacher_answer, temp_dir = temp_dir)
        print("Vision model responded")
    fb = json.loads(response)
    print("fb-------------->",fb)
    return fb

def process_chunk(questions_chunk, answers_chunk, analyze_text_function, analyze_vision_function, student_answers, temp_pdf_path, temp_dir):
    try:
       
        print("Questions chunk given to model----->",questions_chunk)
        print("-------------------------")
        print("answers chunk given to model---------->",answers_chunk)

        #process_fn = partial(process_question, analyze_function, headers, temp_pdf_path=temp_pdf_path, temp_dir=temp_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
               
            futures = []
            for question, teacher_answer in zip(questions_chunk, answers_chunk):
                futures.append(executor.submit(process_question, analyze_text_function,analyze_vision_function, question, teacher_answer, student_answers, temp_pdf_path, temp_dir,))
            #futures = executor.map(process_fn, questions_chunk, answers_chunk)

            feedback =[]
            for future in concurrent.futures.as_completed(futures):
                feedback.append(future.result())
                
        return feedback
    except Exception as e:
         st.write(logger.error("Error in process_chunk function"))
         logger.exception("Exception occurred: %s", e)
         print("Got error in process_chunk function")
         st.write("got error in process_chunk function")
         st.write(e)

def reset_app_state(list_of_questions,listofanswers):
    st.cache_data.clear()
    if os.path.exists(list_of_questions):
        os.remove(list_of_questions)
    if os.path.exists(listofanswers):
         os.remove(listofanswers)
    st.success("Cache cleared, Now you can evaluate with different question paper!")
    
if __name__ == "__main__":
    st.title("Paper Evaluation with Teachers Solutions using GPT-4")
    st.sidebar.title("Upload Files")
    questions_pdf = st.sidebar.file_uploader("**Upload Questions**", type=['pdf'])
    Teachers_answers_pdf = st.sidebar.file_uploader("**Upload Teacher's Answers**", type=['pdf'])
    Students_answers_pdf = st.sidebar.file_uploader("**Upload Student's Answers**", type=['pdf'])

    questions_file = "listofquestions.txt"
    answers_file = "listofanswers.txt"

    checkbox = st.sidebar.checkbox("Check Me to Evaluate Different Question Paper")

    if checkbox:
        reset_app_state(list_of_questions=questions_file,listofanswers=answers_file)

    # Button to trigger evaluation
    if st.button("Evaluate"):
        if questions_pdf and Students_answers_pdf and Teachers_answers_pdf:
            try:
                answers_pdf_bytes = Students_answers_pdf.read()
                temp_dir = tempfile.mkdtemp(prefix="output_images_folder")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        temp_pdf.write(Students_answers_pdf.getbuffer())
                        temp_pdf_path = temp_pdf.name

                questions_file = "listofquestions.txt"
                answers_file = "listofanswers.txt"

                if os.path.exists(questions_file) and os.path.exists(answers_file):
                     try:
                        listofquestions = load_textfile(path=questions_file)
                        listofanswers = load_textfile(path=answers_file)
                        listofquestions = listofquestions.strip().split('\n')
                        listofanswers = listofanswers.strip().split('\n')
                        listofanswers = convert_string_to_dict(listofanswers)
                        logger.info("List of questions and answers found in path")
                     except Exception as e:
                          st.write(e)
                          logger.error("Occured error while loading the questions")
                          logger.exception(e)
                else:
                 
                    Questions_extractor = PDFQuestionExtractor(
                                openai_api_key=api_key,
                                azure_endpoint=azure_endpoint,
                                azure_key=azure_key
                            )
                    listofquestions = Questions_extractor.analyze_read(questions_pdf)
                    

                    # Answers_extractor = PDFAnswersExtractor(
                    #             azure_endpoint=azure_endpoint,
                    #             azure_key= azure_key
                    #         )
                    # answers_with_pagenumbers, answers, sidebysideanswers = Answers_extractor.analyze_read(answers_pdf_bytes)
                    #s_t2 = time.time()
                    extractor = PDFTeacherAnswerExtractor(openai_api_key= api_key,azure_endpoint=azure_endpoint,azure_key= azure_key )
                    listofanswers = extractor.analyze_read(Teachers_answers_pdf)

                #e_t2 = time.time()
               # print(f"Time taken to extract teachers solutions text + list of teachers solutions using OpenAI: {e_t2-s_t2}")
                extractor = PDFStudentAnswerExtractor(openai_api_key= api_key,azure_endpoint=azure_endpoint,azure_key= azure_key )
                studentanswers = extractor.analyze_read(answers_pdf_bytes)

                
                def save_questions_to_file(questions, file_path):
                    with open(file_path, "w",encoding='utf-8') as file:
                        for question in questions:
                            file.write(f"{question}\n")
            # Assuming you have the necessary imports and definitions for VisionModelAnalyzer and analyze_answers_text_model

                chunk_size = 8
                question_chunks = chunkify(listofquestions, n=chunk_size)
                print("Length of Questions_Chunks------------->",len(question_chunks))
                answer_chunks = chunkify(listofanswers, chunk_size)
                print("Lenth of Teachers_Answer Chunks----------->",len(answer_chunks))
                    
                filename = 'Evluation_sheet3_with_parallelprocessing.txt'
                evaluation_sheet = []
                for questions_chunk, answers_chunk in zip(question_chunks, answer_chunks):
                    results = process_chunk(questions_chunk, answers_chunk, analyze_answers_text_model, analyze_answers_vision_model, studentanswers, temp_pdf_path, temp_dir)
                    print("Results----------->",results)
                    un_sorted = sort_questions(results)
                    for result in un_sorted:  
                        st.write(result)
                        evaluation_sheet.append(result)
                    save_questions_to_file(evaluation_sheet,file_path=filename)
                    logger.info(f"Questions saved to {filename}")  

                st.write("Paper Evaluation is done ✅")
                st.write("Skip the Checkbox at sidebar, if you want to evalute another student answers for the same question paper")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        else:
            st.warning("Please upload both questions and answers files.")



