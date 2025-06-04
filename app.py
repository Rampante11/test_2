from flask import Flask, render_template, request, jsonify, send_file
import os
import re
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
import faiss
from openai import OpenAI
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from typing import Dict
from flask_cors import CORS
from werkzeug.utils import secure_filename
from io import BytesIO
import pdfplumber
import json





app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'rag-dataset'))
CORS(app, origins=[
    "https://rag-facturas.onrender.com",           # Si está todo en el mismo proyecto
    "http://localhost:5000"
])# Funciones aquí:


# Añade esto JUSTO DEBAJO de la configuración
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.chmod(app.config['UPLOAD_FOLDER'], 0o755)
        print(f"✅ Directorio creado: {app.config['UPLOAD_FOLDER']}")
    except PermissionError as e:
        print(f"❌ Error de permisos: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        exit(1)

def procesamiento_pdf():

    # Configuración
    DATA_DIR = "rag-dataset"
    CHUNKS_DIR = "chunks_storage"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def clean_text(text: str) -> str:
        """Limpia texto de artefactos de PDF"""
        text = re.sub(r'file:///.*?\[\d+/\d+/\d+.*?\]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = ' '.join(text.split())
        return text.strip()

    def process_pdfs():
        """Procesa PDFs y guarda chunks en disco"""
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Directorio {DATA_DIR} no encontrado")

        os.makedirs(CHUNKS_DIR, exist_ok=True)

        # Encontrar todos los PDFs
        pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
                    if f.endswith('.pdf')]

        if not pdf_files:
            raise ValueError(f"No se encontraron PDFs en {DATA_DIR}")

        print(f"Procesando {len(pdf_files)} archivos PDF...")

        all_chunks = []
        for pdf in pdf_files:
            try:
                loader = PyMuPDFLoader(pdf)
                pages = loader.load()

                # Limpiar y dividir
                for page in pages:
                    page.page_content = clean_text(page.page_content)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=lambda x: len(x.split())
                )

                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                print(f"✅ {os.path.basename(pdf)} → {len(chunks)} chunks")

            except Exception as e:
                print(f"❌ Error en {os.path.basename(pdf)}: {str(e)}")
                continue

        # Guardar chunks
        with open(f"{CHUNKS_DIR}/document_chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)

        print(f"\n✅ Chunks guardados en {CHUNKS_DIR}/document_chunks.pkl")
        print(f"Total chunks creados: {len(all_chunks)}")
        print(f"Ejemplo del primer chunk:\n{'-'*50}")
        print(all_chunks[0].page_content)


    # Configuración
    CHUNKS_DIR = "chunks_storage"
    EMBEDDINGS_DIR = "embeddings_storage"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo local eficiente

    def generate_embeddings():
        """Genera y guarda embeddings a partir de chunks"""
        if not os.path.exists(f"{CHUNKS_DIR}/document_chunks.pkl"):
            raise FileNotFoundError("No se encontraron chunks procesados")

        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

        # Cargar chunks
        with open(f"{CHUNKS_DIR}/document_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        print(f"\nCargados {len(chunks)} chunks para generar embeddings...")

        # Cargar modelo de embeddings
        print("Cargando modelo de embeddings...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        # Generar embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True)

        # Crear y guardar índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        faiss.write_index(index, f"{EMBEDDINGS_DIR}/index.faiss")

        # Guardar metadata
        metadata = {
            "chunks": chunks,
            "embedding_model": EMBEDDING_MODEL,
            "dimension": dimension
        }

        with open(f"{EMBEDDINGS_DIR}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"\n✅ Embeddings guardados en {EMBEDDINGS_DIR}")
        print(f"Dimensión de embeddings: {dimension}")
        print(f"Tamaño del índice: {index.ntotal} vectores")

    process_pdfs()
    generate_embeddings()


def crear_excel():
    # Configuración
    DATA_DIR = "rag-dataset"
    LLM_MODEL = "meta-llama/llama-3-70b-instruct"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Cliente LLM
    llm_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # Template para extracción
    EXTRACTION_TEMPLATE = """Analiza el siguiente texto de factura y extrae los siguientes campos en formato JSON:
    - numero_factura (combina punto de venta y número de comprobante si es necesario)
    - cae (código de autorización electrónica)
    - vencimiento_cae (fecha de vencimiento del CAE)
    - cuit_emisor (CUIT del emisor)
    - total (importe total en formato numérico)
    - fecha_emision (fecha de emisión del documento)
    - razon_social (nombre del emisor)
    - Concepto_facturado (Mes facturado)

    Texto de la factura:
    {text}

    Devuelve SOLO el JSON, sin comentarios adicionales. Ejemplo:
    {{
    "numero_factura": "0003-00000568",
    "cae": "75206077555680",
    "vencimiento_cae": "23/05/2025",
    "cuit_emisor": "30-71097330-6",
    "total": 6518093.93,
    "fecha_emision": "13/05/2025",
    "razon_social": "TERMAIR SRL",
    "Concepto_facturado": "Abril 2025"
    }}"""

    # Función para extracción de texto
    def extract_text_from_pdf(pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error al leer PDF {pdf_path}: {str(e)}")
        return text

    # Función para extraer datos con LLM
    def extract_invoice_data_with_rag(text: str) -> Dict:
        try:
            prompt = EXTRACTION_TEMPLATE.format(text=text[:10000])
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            json_str = response.choices[0].message.content.strip()
            json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            print(f"Error en extracción RAG: {str(e)}")
            return {}

    # Procesar PDFs y generar Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Facturas"

    headers = [
        "Archivo PDF", "Número Factura", "CAE", "Vencimiento CAE",
        "CUIT Emisor", "Total ($)", "Razón Social", "Fecha Emisión",
        "Días al Vencimiento", "Concepto Facturado"
    ]
    ws.append(headers)

    for col in ws[1]:
        col.font = Font(bold=True)
        col.alignment = Alignment(horizontal='center')

    total_procesadas = 0

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"\nProcesando: {filename}")
            text = extract_text_from_pdf(filepath)
            if not text:
                print(f"⚠️ No se pudo extraer texto de {filename}")
                continue

            invoice_data = extract_invoice_data_with_rag(text)
            if not invoice_data:
                print(f"⚠️ No se extrajeron datos de {filename}")
                continue

            dias_restantes = "N/D"
            if 'vencimiento_cae' in invoice_data:
                try:
                    vencimiento = datetime.strptime(invoice_data['vencimiento_cae'], "%d/%m/%Y")
                    dias_restantes = max(0, (vencimiento - datetime.now()).days)
                except:
                    pass

            ws.append([
                filename,
                invoice_data.get('numero_factura', 'N/D'),
                invoice_data.get('cae', 'N/D'),
                invoice_data.get('vencimiento_cae', 'N/D'),
                invoice_data.get('cuit_emisor', 'N/D'),
                invoice_data.get('total', 0),
                invoice_data.get('razon_social', 'N/D'),
                invoice_data.get('fecha_emision', 'N/D'),
                dias_restantes,
                invoice_data.get('Concepto_facturado', 'N/D')
            ])
            total_procesadas += 1
            print(f"✅ Datos extraídos: {invoice_data}")

    # Ajustar anchos de columnas
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        ws.column_dimensions[column_letter].width = max_length + 2

    print(f"\n🎉 Total facturas procesadas: {total_procesadas}")
    return wb



# Configuración RAG (ajusta paths según tu estructura real)
EMBEDDINGS_DIR = "embeddings_storage"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek/deepseek-r1:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Componentes RAG globales
rag_components = {
    "index": None,
    "chunks": None,
    "embedder": None,
    "llm_client": None
}

def initialize_rag_system():
    global rag_chain

    """Carga todos los componentes necesarios para el RAG"""
    try:
        # Cargar modelo de embeddings
        rag_components["embedder"] = SentenceTransformer(EMBEDDING_MODEL)

        # Cargar índice FAISS y metadatos
        if not all(os.path.exists(f"{EMBEDDINGS_DIR}/{f}") for f in ["index.faiss", "metadata.pkl"]):
            raise FileNotFoundError("Ejecuta primero el procesamiento de PDFs")

        rag_components["index"] = faiss.read_index(f"{EMBEDDINGS_DIR}/index.faiss")
        with open(f"{EMBEDDINGS_DIR}/metadata.pkl", "rb") as f:
            data = pickle.load(f)
            rag_components["chunks"] = data if isinstance(data, list) else data.get("chunks", [])

        # Configurar cliente LLM
        rag_components["llm_client"] = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

        print("✅ Sistema RAG inicializado correctamente")

    except Exception as e:
        print(f"❌ Error inicializando RAG: {str(e)}")

def rag_chain(question: str):
    """Pipeline completo de RAG"""
    if not rag_components["index"] or not rag_components["chunks"]:
        return "El sistema no está listo. Primero procesa los PDFs."

    try:
        # Búsqueda semántica
        query_embedding = rag_components["embedder"].encode([question])
        query_array = np.array(query_embedding).astype('float32').reshape(1, rag_components["index"].d)
        _, indices = rag_components["index"].search(query_array, 3)
        relevant_docs = [rag_components["chunks"][i] for i in indices[0] if i < len(rag_components["chunks"])]

        if not relevant_docs:
            return "No se encontraron documentos relevantes."

        # Generar contexto
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Generar respuesta
        response = rag_components["llm_client"].chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Responde en español de forma COMPLETA usando SOLO esta información:

                CONTEXTO:
                {context}

                PREGUNTA: {question}

                INSTRUCCIONES:
                - Proporciona una respuesta detallada
                - Incluye todos los datos relevantes
                - Usa el formato: ': [texto completo]'"""
            }],
            max_tokens=1500,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error en el sistema: {str(e)}"





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json(force=True)  # <- fuerza el parseo
        print(f"📦 Datos recibidos en /chat: {data}")

        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Mensaje vacío'}), 400

        respuesta = rag_chain(user_message)
        print(f"💡 Respuesta generada: {respuesta}")
        return jsonify({'response': respuesta})

    except Exception as e:
        print(f"\n❌ Error en chat: {str(e)}")  # Log de error
        return jsonify({'error': str(e)}), 500

# Inicializar el sistema al arrancar la app
with app.app_context():
    initialize_rag_system()




@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Verificar si se enviaron archivos
        if 'file' not in request.files:
            return jsonify({"error": "No se detectaron archivos en la solicitud"}), 400

        files = request.files.getlist('file')
        if not files:
            return jsonify({"error": "La lista de archivos está vacía"}), 400

        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue

            # Validar y guardar
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filename)
            else:
                return jsonify({"error": f"Tipo de archivo no permitido: {file.filename}"}), 400

        return jsonify({
            "success": True,
            "message": f"Subidos {len(uploaded_files)} archivos",
            "files": uploaded_files
        })

    except Exception as e:
        app.logger.error(f"ERROR en /upload: {str(e)}")
        return jsonify({"error": "Fallo en el servidor. Contacta al administrador."}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf'}  # Solo PDFs





@app.route('/procesar_pdf', methods=['POST'])
def procesar_pdf_route():
    try:
        procesamiento_pdf()
        global rag_chain  # Asegúrate de declararlo como global si no lo está
        initialize_rag_system()
        return jsonify({'message': 'Procesamiento PDF completado'})
    except Exception as e:
        import traceback
        traceback.print_exc()  # Esto mostrará el error completo en la consola
        return jsonify({'error': str(e)}), 500





@app.route('/crear_excel', methods=['POST'])
def crear_excel_route():
    try:
        # Generamos el Workbook en memoria
        output = BytesIO()
        wb = crear_excel()  # Modifica crear_excel() para que devuelva el objeto Workbook
        wb.save(output)
        output.seek(0)

        # Enviamos el fichero
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Reporte_Facturas_RAG_{timestamp}.xlsx"
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    # Configuración para producción
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Render
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False en producción