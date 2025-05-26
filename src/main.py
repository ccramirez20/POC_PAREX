import os
import json
import uuid
from typing import List, Dict, Any
import tempfile
from openai import OpenAI
import openai
import wikipedia
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pdfminer.high_level import extract_text as extract_pdf_text
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import io
import cv2
import numpy as np
import fitz  # PyMuPDF para manejo avanzado de PDFs

import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

print(">> Ejecutando archivo:", __file__)

load_dotenv()

ENV_API_KEY = "OPENAI_API_KEY"
if ENV_API_KEY not in os.environ:
    raise RuntimeError(f"Debe establecer la variable de entorno {ENV_API_KEY}")
openai.api_key = os.getenv(ENV_API_KEY)

MODEL = "gpt-3.5-turbo"

client = OpenAI(api_key=os.getenv(ENV_API_KEY))


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

app = FastAPI(title="TalentAI", version="1.0")

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

# --- NUEVO: contexto global único para chat ---
global_chat_history: List[Dict[str, str]] = []
all_cvs_context: List[Dict[str, Any]] = []

def extraer_texto_pdf(path: str) -> str:
    try:
        # Intentar extracción directa primero
        texto = extract_pdf_text(path)
        
        # Si hay texto suficiente, devolverlo
        if texto and len(texto.strip()) > 50:
            return texto
            
        # Si no hay texto o es muy poco, intentar extraer como imágenes
        print("PDF parece contener principalmente imágenes, intentando OCR...")
        
        import fitz  # PyMuPDF
        texto_completo = ""
        doc = fitz.open(path)
        
        # Crear un directorio temporal específico para este procesamiento
        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Escala x2 para mejor OCR
                
                # Guardar en un archivo temporal dentro del directorio temporal
                temp_img_path = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(temp_img_path)
                
                # Cargar con PIL para el procesamiento
                img = Image.open(temp_img_path)
                
                # Preprocesar y aplicar OCR
                img_mejorada = mejorar_imagen(img)
                texto_pagina = pytesseract.image_to_string(img_mejorada, lang='spa+eng')
                
                # Si no hay suficiente texto, intentar con la imagen original
                if len(texto_pagina.strip()) < 10:
                    texto_pagina = pytesseract.image_to_string(img, lang='spa+eng')
                    
                texto_completo += f"\n--- Página {page_num+1} ---\n{texto_pagina}"
                
                # No es necesario eliminar los archivos manualmente, 
                # tempfile.TemporaryDirectory se encarga de ello
        
        if len(texto_completo.strip()) > 50:
            return texto_completo
        else:
            return "PDF parece estar vacío o contiene solo imágenes de baja calidad. Intente subir una versión de mejor calidad."
            
    except Exception as e:
        print(f"Error al procesar PDF: {str(e)}")
        return f"Error al procesar PDF: {str(e)}"

def mejorar_imagen(image):
    """Aplica técnicas de preprocesamiento para mejorar el OCR"""
    try:
        # Convertir a escala de grises si es necesario
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convertir a formato numpy para procesamiento con OpenCV
        img_np = np.array(image)
        
        # Aplicar umbral adaptativo para mejorar contraste
        # Verificar que la imagen sea válida para umbral adaptativo
        if img_np.size == 0 or img_np.min() == img_np.max():
            return image  # Devolver imagen original si no es válida
        
        # Asegurarse de que la imagen tenga el tipo de datos correcto
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        
        try:
            # Aplicar umbral adaptativo
            img_np = cv2.adaptiveThreshold(
                img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Reducir ruido
            img_np = cv2.medianBlur(img_np, 3)
        except cv2.error:
            # Si falla el procesamiento de OpenCV, retornar imagen original
            return image
        
        # Convertir de nuevo a formato PIL
        return Image.fromarray(img_np)
    except Exception as e:
        print(f"Error al mejorar imagen: {e}")
        return image  # Devolver imagen original en caso de error

def extraer_texto_imagen(path: str) -> str:
    """Extrae texto de una imagen con preprocesamiento avanzado"""
    try:
        # Cargar la imagen
        image = Image.open(path)
        
        # Guardar dimensiones originales para verificar si es demasiado pequeña
        width, height = image.size
        if width < 300 or height < 300:
            # Escalar imagen si es muy pequeña
            factor = max(300/width, 300/height)
            new_size = (int(width * factor), int(height * factor))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Aplicar preprocesamiento para mejorar calidad
        imagen_mejorada = mejorar_imagen(image)
        
        # Intentar con la imagen mejorada
        texto_mejorado = pytesseract.image_to_string(imagen_mejorada, lang='spa+eng')
        
        # Revisar si tenemos resultados
        if texto_mejorado.strip():
            return texto_mejorado
            
        # Si no hay resultados, intentar con la imagen original
        texto_original = pytesseract.image_to_string(image, lang='spa+eng')
        
        # Si ninguna de las dos opciones funcionó
        if not texto_original.strip() and not texto_mejorado.strip():
            return "No se pudo extraer texto de la imagen. El documento puede estar en blanco o ser ilegible. Intente con una imagen de mayor resolución."
            
        # Devolver el texto más largo (probablemente el mejor)
        return texto_mejorado if len(texto_mejorado) > len(texto_original) else texto_original
        
    except Exception as e:
        print(f"Error al procesar imagen: {str(e)}")
        return f"Error al procesar la imagen: {str(e)}"

def parsear_cv(texto: str) -> Dict[str, Any]:
    system_msg = (
        "Eres un experto en RRHH y formateas CVs. "
        "Recibe texto de un CV y extrae en JSON las claves: "
        "\u2022 work_experience: lista de objetos con campos title, company, start_date, end_date. "
        "\u2022 education: lista de objetos con fields degree, institution, start_date, end_date. "
        "\u2022 languages: lista de objetos language y proficiency. "
        "\u2022 skills: lista de cadenas. "
        "Devuelve únicamente un JSON válido, sin explicaciones, encabezados ni texto adicional."
    )
    user_msg = f"Texto del CV:\n{texto}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=1500
    )

    content = response.choices[0].message.content.strip()

    if not content:
        raise ValueError("La respuesta del modelo está vacía")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("Error al decodificar JSON de OpenAI:", content)
        raise ValueError("La respuesta del modelo no es un JSON válido") from e

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
        "Comparas un perfil de candidato con una descripción de puesto y devuelves en JSON: "
        "compatibility_percentage (0-100), strengths (fortalezas), gaps (brechas)."
    )
    user_msg = (
        f"Perfil del candidato (JSON): {json.dumps(cv, ensure_ascii=False)}\n"
        f"Descripción del puesto (JSON): {json.dumps(descripcion_puesto, ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=800
    )
    return json.loads(response.choices[0].message.content)

@app.post("/procesar_cvs")
async def procesar_cvs(cvs: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
    resultados = []
    for archivo in cvs:
        try:
            contenido = await archivo.read()
            ext = os.path.splitext(archivo.filename)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(contenido)
                tmp_path = tmp.name
            
            try:
                if ext == ".pdf":
                    texto = extraer_texto_pdf(tmp_path)
                elif ext in [".png", ".jpg", ".jpeg"]:
                    texto = extraer_texto_imagen(tmp_path)
                else:
                    texto = contenido.decode('utf-8', errors='replace')
                
                # Verificar si el texto contiene mensajes de error conocidos
                mensajes_error = [
                    "No se pudo extraer texto",
                    "Error al procesar",
                    "parece estar vacío"
                ]
                
                # Si es un mensaje de error claro, lo propagamos
                if any(msg in texto for msg in mensajes_error) and len(texto) < 200:
                    raise ValueError(texto)
                
                # Verificar si el texto está realmente vacío o es demasiado corto
                if not texto or len(texto.strip()) < 50:
                    raise ValueError("El texto extraído es insuficiente para procesar el CV")
                
                # Si llegamos aquí, tenemos texto suficiente para procesar
                print(f"Texto extraído exitosamente de {archivo.filename}: {len(texto)} caracteres")
                
                try:
                    cv_info = parsear_cv(texto)
                    
                    # Verificación básica del resultado del parsing
                    if not cv_info or not any(key in cv_info for key in ['work_experience', 'education', 'skills']):
                        raise ValueError("El CV no contiene secciones básicas reconocibles")
                        
                    empresas = [e.get('company') for e in cv_info.get('work_experience', [])]
                    cv_info['verificacion_empresas'] = verificar_empresas(empresas)
                    compat = calcular_compatibilidad(cv_info)

                    # Guardar en contexto global
                    all_cvs_context.append({
                        "filename": archivo.filename,
                        "cv_info": cv_info,
                        "compatibilidad": compat
                    })

                    # Actualizar global_chat_history sistema con nuevo CV y contexto
                    resumen_cvs = json.dumps(all_cvs_context, ensure_ascii=False)
                    system_msgs = [
                        {"role": "system", "content": "Eres un asistente de talento humano que considera todos los CVs procesados. Da respuestas útiles y concretas y da recomendaciones."},    
                        {"role": "system", "content": f"Contexto global de CVs: {resumen_cvs}"}
                    ]
                    global_chat_history.clear()
                    global_chat_history.extend(system_msgs)

                    resultados.append({
                        "session_id": "global",
                        "filename": archivo.filename,
                        "cv_info": cv_info,
                        "compatibilidad": compat
                    })
                except Exception as parsing_error:
                    # Error específico en el parsing del CV
                    raise ValueError(f"Error al analizar el contenido del CV: {str(parsing_error)}")
                    
            except Exception as e:
                print(f"Error al procesar {archivo.filename}: {str(e)}")
                resultados.append({
                    "filename": archivo.filename,
                    "error": f"No se pudo procesar este CV: {str(e)}"
                })
            finally:
                # Asegurarnos de que eliminamos el archivo temporal
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            print(f"Error general con {archivo.filename}: {str(e)}")
            resultados.append({
                "filename": archivo.filename,
                "error": f"Error al procesar el archivo: {str(e)}"
            })
            
    return resultados


class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Chat que utiliza contexto global para responder,
    ignorando session_id y siempre usando la misma sesión global.
    """
    if request.session_id != "global":
        # Podemos advertir que el session_id es ignorado, o forzar a "global"
        pass

    if not global_chat_history:
        # Si aún no hay CVs, inicializar sistema básico
        global_chat_history.extend([
            {"role": "system", "content": "Eres un asistente de talento humano sin datos de CV. Da respuestas útiles y concretas así como recomendaciones."}
        ])

    # Agregar mensaje usuario
    global_chat_history.append({"role": "user", "content": request.message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=global_chat_history,
        temperature=0.7,
        max_tokens=100
    )

    reply = response.choices[0].message.content
    # Agregar respuesta asistente al historial
    global_chat_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)