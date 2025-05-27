import os
import json
import uuid
import tempfile
import io
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
import PyPDF2  

load_dotenv()

print(">> Ejecutando archivo:", __file__)

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

app.mount("/static", StaticFiles(directory="src/templates", html=True), name="static")

# Ruta para servir el archivo HTML principal
@app.get("/", response_class=FileResponse)
async def get_index():
    return FileResponse("src/templates/index.html")

# Contexto global de chat
global_chat_history: List[Dict[str, str]] = []
all_cvs_context: List[Dict[str, Any]] = []

def extraer_texto_pdf(contenido: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(contenido))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Error extrayendo texto del PDF: {e}")
        return ""

def parsear_cv(texto: str) -> Dict[str, Any]:
    system_msg = (
        "Eres un experto en RRHH y formateas CVs. "
        "Recibe texto de un CV y extrae en JSON las claves: "
        "• work_experience: lista de objetos con campos title, company, start_date, end_date. "
        "• education: lista de objetos con fields degree, institution, start_date, end_date. "
        "• languages: lista de objetos language y proficiency. "
        "• skills: lista de cadenas. "
        "Devuelve únicamente un JSON válido, sin explicaciones, encabezados ni texto adicional."
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
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parseando JSON de OpenAI: {content}")
        raise ValueError("La respuesta de OpenAI no es un JSON válido.")

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
        "Compara un perfil de candidato con una descripción de puesto y responde SOLO con un JSON válido con las siguientes claves: "
        "compatibility_percentage (entero 0-100), strengths (lista de cadenas), gaps (lista de cadenas). "
        "No incluyas explicaciones, encabezados, ni texto adicional. SOLO el JSON."
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
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parseando compatibilidad JSON de OpenAI: {content}")
        raise ValueError("La respuesta de OpenAI (compatibilidad) no es un JSON válido.")

@app.post("/procesar_cvs")
async def procesar_cvs(cvs: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    resultados = []
    for archivo in cvs:
        try:
            contenido = await archivo.read()
            texto = extraer_texto_pdf(contenido)
            print(f"Texto extraído: {texto[:200]}")  # Solo los primeros 200 caracteres

            if not texto or len(texto.strip()) < 50:
                raise ValueError("Texto insuficiente para procesar el CV")

            cv_info = parsear_cv(texto)
            print(f"CV parseado: {cv_info}")

            empresas = [e.get('company') for e in cv_info.get('work_experience', [])]
            cv_info['verificacion_empresas'] = verificar_empresas(empresas)
            compat = calcular_compatibilidad(cv_info)
            print(f"Compatibilidad: {compat}")

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
            print(f"Error procesando {archivo.filename}: {e}")
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
            {"role": "system", "content": "Eres un asistente de talento humano. Responde siempre de forma breve, clara y concisa. Usa frases cortas."}
        ])
    else:
        global_chat_history.extend([
            {"role": "system", "content": "Eres un asistente de talento humano. Considera los CVs procesados y responde SIEMPRE de forma breve, clara y concisa. Usa frases cortas y directas. Evita explicaciones largas."}
        ])

    global_chat_history.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=global_chat_history,
        temperature=0.7,
        max_tokens=200
    )

    reply = response.choices[0].message.content
    global_chat_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
