import os
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import vision
import tempfile
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import toml
import tempfile

load_dotenv()

groq_config = st.secrets["groq"]
client = Groq(
    api_key=groq_config["api_key"],
)

# Extract Google credentials from secrets
google_credentials = secrets["google_application_credentials"]

# Create a temporary file for Google credentials
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
    # Convert credentials dictionary to JSON format
    tmp_file.write(toml.dumps(google_credentials).encode())
    tmp_file_path = tmp_file.name

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = tmp_file_path

# Initialize Google Vision Client
def init_vision_client():
    try:
        client = vision.ImageAnnotatorClient()
        return client
    except Exception as e:
        st.error(f"Error initializing Google Vision API client: {e}")
        return None

# Extract text from an image using Google Vision API
def extract_text_from_image(image_path, client):
    try:
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.error.message:
            raise Exception(response.error.message)
        texts = response.text_annotations
        return texts[0].description if texts else ""
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

# Extract text from a PDF using Google Vision API
def extract_text_from_pdf(pdf_path, client):
    try:
        # Convert PDF pages to images
        with tempfile.TemporaryDirectory() as path:
            images = convert_from_path(pdf_path, dpi=300, output_folder=path, fmt='png')
            extracted_text = ""
            for i, image in enumerate(images):
                # Save image to a temporary file
                temp_image_path = os.path.join(path, f"page_{i + 1}.png")
                image.save(temp_image_path, 'PNG')
                # Extract text from the image
                page_text = extract_text_from_image(temp_image_path, client)
                extracted_text += f"--- Page {i + 1} ---\n{page_text}\n\n"
            return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    
# Process Excel/CSV files
def process_excel(file):
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        text = df.to_string(index=False)
        return text
    except Exception as e:
        st.error(f"Error processing Excel/CSV file: {e}")
        return ""

# Interact with GPT-4
def ask_gpt4(prompt):
    try:
        response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
    )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"Error communicating with GPT: {e}")
        return ""

# Streamlit App
def main():
    st.title("Multi-Input Groq Chatbot")
    st.write("Upload a text file, image, PDF, Excel, or CSV file and ask questions about its content.")

    # Initialize clients and keys
    vision_client = init_vision_client()

    # Sidebar for input options
    st.sidebar.header("Input Options")
    input_type = st.sidebar.selectbox("Select input type", ["Text", "Image", "PDF", "Excel/CSV"])

    # Initialize session state for history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    extracted_text = ""

    if input_type == "Text":
        st.header("Enter Text")
        user_text = st.text_area("Input Text", height=200)
        if user_text:
            extracted_text = user_text

    elif input_type == "Image":
        st.header("Upload Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        if uploaded_image and vision_client:
            try:
                # Save the uploaded image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[1]) as tmp_image:
                    tmp_image.write(uploaded_image.read())
                    tmp_image_path = tmp_image.name

                # Extract text
                extracted_text = extract_text_from_image(tmp_image_path, vision_client)
                st.subheader("Extracted Text")
                st.text_area("Text from Image", value=extracted_text, height=200)
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif input_type == "PDF":
        st.header("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_pdf and vision_client:
            try:
                # Save the uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_pdf.read())
                    tmp_pdf_path = tmp_pdf.name

                # Extract text
                with st.spinner("Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(tmp_pdf_path, vision_client)

                st.subheader("Extracted Text")
                st.text_area("Text from PDF", value=extracted_text, height=300)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    elif input_type == "Excel/CSV":
        st.header("Upload Excel/CSV")
        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])
        if uploaded_file:
            extracted_text = process_excel(uploaded_file)
            st.subheader("Extracted Text")
            st.text_area("Text from Excel/CSV", value=extracted_text, height=300)

 # Ask a question about the extracted text
    if extracted_text:
        st.header("Ask a Question")
        question = st.text_area("Enter your question about the content above:",height=300)

        if st.button("Get Answer") and question:
            with st.spinner("Generating answer..."):
                # Build prompt with conversation history
                prompt = "Here is the content:\n\n" + extracted_text
                if st.session_state.conversation_history:
                    prompt += "\n\nPrevious conversation:\n" + "\n".join(st.session_state.conversation_history)
                prompt += f"\n\nQuestion: {question}\nAnswer:"

                answer = ask_gpt4(prompt)

                # Save the current Q&A to the session state
                st.session_state.conversation_history.append(f"Q: {question}\nA: {answer}")

                st.subheader("Answer")
                st.write(answer)

    # Display conversation history
    if st.session_state.conversation_history:
        st.header("Conversation History")
        st.write("\n".join(st.session_state.conversation_history))

if __name__ == "__main__":
    main()
