import os
import json
import uuid
import tempfile
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import openai
import wikipedia

load_dotenv()

# Configuración OpenAI
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Debe establecer la variable de entorno OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)
MODEL = "gpt-4-turbo"

# Descripción del puesto para evaluación
descripcion_puesto: Dict[str, Any] = {
    "titulo": "Ingeniero de Software Senior",
    "responsabilidades": [
        "Diseñar e implementar sistemas backend escalables.",
        "Colaborar con equipos multifuncionales.",
        "Mantener estándares de calidad y pruebas automatizadas."
    ],
    "requisitos": [
        "5+ años de experiencia en Python y Django.",
        "Conocimiento de Docker y Kubernetes.",
        "Nivel avanzado de inglés."
    ]
}

# Inicialización FastAPI
app = FastAPI(title="TalentAI", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse("templates/index.html")

# Contexto global de chat
global_chat_history: List[Dict[str, str]] = []
all_cvs_context: List[Dict[str, Any]] = []

def extraer_texto_con_openai(contenido: bytes, filename: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            file_upload = client.files.create(file=f, purpose="assistants")
            file_id = file_upload.id

        respuesta = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en extraer texto de documentos. Extrae todo el contenido visible del archivo PDF o imagen."
                },
                {
                    "role": "user",
                    "content": f"Extrae el texto completo de este archivo: {filename}",
                    "file_ids": [file_id]
                }
            ],
            temperature=0.2,
            max_tokens=2000
        )

        texto = respuesta.choices[0].message.content.strip()
        return texto if texto else "No se pudo extraer texto."
    except Exception as e:
        return f"Error al extraer texto con OpenAI: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def parsear_cv(texto: str) -> Dict[str, Any]:
    system_msg = (
        "Eres un experto en RRHH y formateas CVs. Recibe texto de un CV y extrae en JSON las claves: "
        "work_experience (lista de objetos con title, company, start_date, end_date), "
        "education (lista con degree, institution, start_date, end_date), "
        "languages (lista de objetos con language y proficiency), skills (lista de cadenas). "
        "Devuelve solo un JSON válido."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": texto}
        ],
        temperature=0,
        max_tokens=1500
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)

def verificar_empresa(nombre: str) -> bool:
    try:
        wikipedia.page(nombre)
        return True
    except Exception:
        return False

def verificar_empresas(empresas: List[str]) -> Dict[str, bool]:
    return {e: verificar_empresa(e) for e in empresas if e}

def calcular_compatibilidad(cv: Dict[str, Any]) -> Dict[str, Any]:
    system_msg = (
        "Eres un especialista de selección de personal. "
        "Compara un perfil de candidato con una descripción de puesto y devuelve: "
        "compatibility_percentage (0-100), strengths, gaps."
    )
    user_msg = (
        f"Perfil del candidato: {json.dumps(cv, ensure_ascii=False)}\n"
        f"Descripción del puesto: {json.dumps(descripcion_puesto, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return json.loads(response.choices[0].message.content)

@app.post("/procesar_cvs")
async def procesar_cvs(cvs: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    resultados = []
    for archivo in cvs:
        try:
            contenido = await archivo.read()
            texto = extraer_texto_con_openai(contenido, archivo.filename)

            if not texto or len(texto.strip()) < 50:
                raise ValueError("Texto insuficiente para procesar el CV")

            cv_info = parsear_cv(texto)
            empresas = [e.get('company') for e in cv_info.get('work_experience', [])]
            cv_info['verificacion_empresas'] = verificar_empresas(empresas)
            compat = calcular_compatibilidad(cv_info)

            all_cvs_context.append({
                "filename": archivo.filename,
                "cv_info": cv_info,
                "compatibilidad": compat
            })

            resumen_cvs = json.dumps(all_cvs_context, ensure_ascii=False)
            global_chat_history.clear()
            global_chat_history.extend([
                {"role": "system", "content": "Eres un asistente de talento humano que considera todos los CVs procesados. Da respuestas útiles y concretas. Da recomendaciones."}, 
                {"role": "system", "content": f"Contexto global de CVs: {resumen_cvs}"}
            ])

            resultados.append({
                "session_id": "global",
                "filename": archivo.filename,
                "cv_info": cv_info,
                "compatibilidad": compat
            })
        except Exception as e:
            resultados.append({
                "filename": archivo.filename,
                "error": str(e)
            })
    return resultados

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    if not global_chat_history:
        global_chat_history.extend([
            {"role": "system", "content": "Eres un asistente de talento humano sin datos de CV."}
        ])

    global_chat_history.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=global_chat_history,
        temperature=0.7,
        max_tokens=300
    )

    reply = response.choices[0].message.content
    global_chat_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
