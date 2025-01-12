import json
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
import google.generativeai as genai
from requests.exceptions import Timeout
import numpy as np
from scipy.spatial.distance import euclidean, cosine


# Configure Gemini API

genai.configure(api_key="AIzaSyATU4bUjPtCd-x5MRwsmioLBlnYK4Vyd8o")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

# Initialize SentenceTransformer model for embedding
encoder = SentenceTransformer('all-MiniLM-L6-v2') 

# Step 1: Load and parse course data
with open('all_courses_with_study_material.json') as f:
    courses_data = json.load(f)

# Extract course names and descriptions
course_names = list(courses_data.keys())
course_descriptions = [course['data'] for course in courses_data.values()]
course_material = [course['study_material'] for course in courses_data.values()]
# Step 2: Initialize Qdrant client with cloud endpoint and API key

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine(embedding1, embedding2)
# Initialize the Qdrant client
client = QdrantClient(
    url = 'https://07eb4b7a-96aa-4ce9-b054-6892c56fda54.us-west-2-0.aws.cloud.qdrant.io',
    api_key = "-8679vHd9oDwK7iZNwCVmnPc1YV-2IIIMyHeQTDxFimIUWdV7buCYA"
)
#client = QdrantClient(":memory:")
study_material_embedding = encoder.encode(["Give me the study material , links course website etc."])[0]
# Step 6: Query processing function to find similar courses using Qdrant
def search_courses(user_query, timeout=30):
    query_normalized = user_query.lower()
    print(f"Normalized query: '{query_normalized}'")  # Debug print

    queries = query_normalized.split()
    exact_matches = []
    for query in queries:
        exact_matches += [course for course in course_names if query in course.lower()]
    query_embedding = encoder.encode([user_query])[0]
    score = calculate_cosine_similarity(study_material_embedding, query_embedding)
    results = []
    if exact_matches:
        for course in exact_matches:
            idx = course_names.index(course)
            description = course_descriptions[idx]
            response = gemini_model.generate_content("Give only the website link of course {} iitd".format(course))
            description = description + ", the course website : {} , previous year paper links : {}".format(response, " , ".join(course_material[idx]))
            
            results.append({
                "course_name": course,
                "description": description
            })
    else:
        print("No exact match found. Performing semantic search...")  # Debug print
        
        print(f"Query embedding generated: {query_embedding[:10]}...")  # Debug print

        try:
            search_result = client.query_points(
                collection_name="Shipathon",
                query_vector=query_embedding.tolist(),
                limit=10,
            ).points
        except Exception as e:
            print(f"Timeout during search: {e}")
            search_result = []
       
        for hit in search_result:
            description= hit.payload['description']
            idx = course_names.index(hit.payload['course'])
            
            response = gemini_model.generate_content("Give only the website link of course {} iitd".format(hit.payload['course_name']))
            description = description + ", the course website : {} , previous year paper links : {}".format(response, " , ".join(course_material[idx]))
            results.append({
                "course_name": hit.payload['course'],
                "description": description
            })

    if not results:
        return [{"course_name": "No relevant course found", "description": "More information about APL100 is needed."}]
    
    return results
def query(prompt):
  hits = client.query_points(
    collection_name="Shipathon",
    query=encoder.encode(prompt).tolist(),
    limit=10,
    ).points
  final_data=""
  for id,i in enumerate(hits):
    final_data=final_data+"{}. course code:{} , description:{} \n".format(id+1,i.payload["course"],i.payload["description"])
  history = [
    {"role":"user","parts":"suppose the list of courses is\n" +final_data}
  ]

  chat_session = gemini_model.start_chat(
    history=history
  )

  response = chat_session.send_message(prompt)

  return response.text
# Step 7: Generate response using Gemini
def generate_response_with_gemini(user_query, courses_info):
    prompt = f"User query: '{user_query}'\nHere are the top 3 courses that match:\n{courses_info}\nPlease summarize the information and answer the user's query."

    try:
        response = gemini_model.generate_content(prompt) 
        return response.text
    except Timeout as e:
        print(f"Timeout during Gemini response generation: {e}")
        return "The request timed out while generating the response. Please try again."
    except Exception as e:
        print(f"Error during Gemini response generation: {e}")
        return None

# Example usage
import streamlit as st

# Inject custom CSS for styling
def inject_custom_css():
    st.markdown("""
        <style>
            /* Global background */
            .stApp {
                background-color: #121212;
                color: white;
                font-family: 'Arial', sans-serif;
            }

            /* Chat container styling */
            .stChatContainer {
                background-color: #1E1E2F;
                border-radius: 15px;
                padding: 20px;
                max-width: 700px;
                margin: auto;
                margin-top: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            /* User message style (rightmost aligned, adjusts to content) */
            .user-message {
                background-color: #0066cc;
                color: white;
                padding: 12px 16px;
                border-radius: 15px;
                margin-bottom: 10px;
                max-width: 70%;
                display: inline-block;
                text-align: left;
                word-wrap: break-word;
                float: right; /* Align to the right */
                position: relative; /* For positioning the user icon */
            }

            .user-message::before {
                content: "";
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                right: -30px; /* Adjust position as needed */
                width: 20px;
                height: 20px;
                background-color: #0066cc; 
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 12px;
            }

            /* Assistant message style (leftmost aligned, adjusts to content) */
            .assistant-message {
                background-color: #4A4A5A;
                color: white;
                padding: 12px 16px;
                border-radius: 15px;
                margin-bottom: 10px;
                max-width: 70%;
                display: inline-block;
                text-align: left;
                word-wrap: break-word;
                float: left; /* Align to the left */
                position: relative; /* For positioning the assistant icon */
            }

            .assistant-message::before {
                content: "";
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                left: -30px; /* Adjust position as needed */
                width: 20px;
                height: 20px;
                background-color: #4A4A5A; 
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 12px;
            }

            /* Input box styling */
            .stTextInput > div > input {
                background-color: #282c34;
                color: white;
                border: 1px solid #555;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 16px;
                margin-top: 15px;
            }

            /* Input focus */
            .stTextInput > div > input:focus {
                border-color: #0078FF;
                outline: none;
            }
        </style>
    """, unsafe_allow_html=True)

# Inject the CSS
inject_custom_css()

# App title
st.title("Virtual Companion Chat")

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display container
st.markdown('<div class="stChatContainer">', unsafe_allow_html=True)

# Display chat messages from the history
for chat in st.session_state.chat_history:
    role_class = "user-message" if chat["role"] == "user" else "assistant-message"
    st.markdown(f'<div class="{role_class}">{chat["message"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# User input box
user_input = st.chat_input("Type your message...")

# If user sends a message
if user_input:
    # Add user's message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)

    courses_info = search_courses(user_input)

    # Format the courses into a string for the prompt
    courses_info_str = "\n".join([f"Course: {course['course_name']}\nDescription: {course['description']}" for course in courses_info])

    # Generate response using Gemini
    response = generate_response_with_gemini(user_input, courses_info)

    # Generate assistant response
    assistant_response = response  # Replace with AI logic if needed
    st.session_state.chat_history.append({"role": "assistant", "message": assistant_response})
    st.markdown(f'<div class="assistant-message">{assistant_response}</div>', unsafe_allow_html=True)
