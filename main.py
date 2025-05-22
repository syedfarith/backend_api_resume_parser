from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from io import BytesIO
import pdfplumber
import uuid
import docx
from groq import Groq
import cohere
from qdrant_client_setup import setup_qdrant, add_resume_to_qdrant, search_resume
import os
import json

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or "*" for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# Initialize Qdrant
VECTOR_SIZE = 1536  # Cohere model vector size
setup_qdrant(VECTOR_SIZE)


async def extract_text_from_file(file: UploadFile) -> str:
    content = await file.read()

    if file.filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    elif file.filename.endswith(".docx"):
        doc = docx.Document(BytesIO(content))
        text = "\n".join(para.text for para in doc.paragraphs)
        return text
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or DOCX.")


async def parse_resume_to_structured(text: str) -> dict:
    prompt = """
You are a helpful assistant that extracts the following details from a resume text:

Name, Age, Role, Skills (as a list)

Output the result in a strict JSON format with keys: name, age, role, skills,experience,projects, education, certifications, languages, location, contact_info.
Add internship in the experience.
Do not include any other text or explanation. The JSON should be well-structured and valid.

If any information is missing, use null or empty list.

it should be json not a string
give it as direct json don't assign a variable like ' response : 'to it give it as a direct json object
{
    Name: "Name",
    Age: "Age",
    Role: "Role",
    Skills: ["Skill1", "Skill2", "Skill3"]
    etc
}

"""
    client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client_groq.chat.completions.create(
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ],
    model="llama-3.3-70b-versatile",
    stream=False,
)

    result = completion.choices[0].message.content
    return {"response": result}

@app.post("/upload_resume/")
async def upload_resume_file(file: UploadFile = File(...)):
    text = await extract_text_from_file(file)
    structured_data = await parse_resume_to_structured(text)
    print(f"Structured data: {structured_data}")
    print("structured_data",type(structured_data))
    resume_id = str(uuid.uuid4())
    structured_data["id"] = resume_id

    text_to_embed = json.dumps(structured_data)
    response = co.embed(texts=[text_to_embed], input_type="search_document",model="embed-v4.0")
    embedding = response.embeddings[0]
    print(f"Embedding vector length: {len(embedding)}")

    # Create and store payload
    resume_id = str(uuid.uuid4())
    
    add_resume_to_qdrant(embedding, structured_data)

    return {"message": "Resume stored successfully", "resume_id": resume_id}


@app.post("/search/")
async def search_resumes(query: str = Form(...)):
    try:
        # Get query embedding
        response = co.embed(texts=[query], input_type="search_query",model="embed-v4.0")
        query_embedding = response.embeddings[0]

        # Perform search
        results = search_resume(query_embedding)
        print(f"Search results: {results}")
        return [
            {
                "score": result.score,
                 "response": result.payload.get("response") or result.payload.get("text") or result.payload.get("name")

            }
            for result in results
        ]
    except Exception as e:
        return {"error": str(e)}
