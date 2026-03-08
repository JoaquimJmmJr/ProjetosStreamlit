import os
from io import BytesIO

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()


# ============================ Estado ============================ #

if "optimized_cv" not in st.session_state:
    st.session_state.optimized_cv = ""

if "last_output" not in st.session_state:
    st.session_state.last_output = ""


# ============================ Fontes ============================ #

FONT_FILES = {
    "Helvetica": None,
    "Montserrat": "fonts/Montserrat-Regular.ttf",
    "Merriweather": "fonts/Merriweather-Regular.ttf",
    "Playfair Display": "fonts/PlayfairDisplay-Regular.ttf",
    "Yanone Kaffeesatz": "fonts/YanoneKaffeesatz-Regular.ttf",
    "Touvlo": "fonts/Touvlo-Regular.ttf",
}


# ============================ Helpers ============================ #

def get_secret(key: str):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = 200_000) -> str:
    """Extrai texto de PDF com PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    return text[:max_chars] if text else ""


def get_llm(model_choice: str):
    """Cria o LLM (LlamaIndex puro)."""
    if model_choice == "Gemini (Google)":
        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY não encontrado em st.secrets nem no .env")
            st.stop()

        os.environ["GOOGLE_API_KEY"] = api_key
        return Gemini(model="models/gemini-2.5-flash")

    if model_choice == "Groq (Llama 4 Maverick)":
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY não encontrado em st.secrets nem no .env")
            st.stop()

        os.environ["GROQ_API_KEY"] = api_key
        return Groq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
        )

    st.error("Modelo inválido.")
    st.stop()


def llm_complete(llm, prompt: str) -> str:
    """Wrapper robusto para pegar o texto retornado pelo LlamaIndex."""
    resp = llm.complete(prompt)
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    return str(resp).strip()


def extract_optimized_cv(full_output: str) -> str:
    start = "===CURRICULO_OTIMIZADO_INICIO==="
    end = "===CURRICULO_OTIMIZADO_FIM==="

    if start in full_output and end in full_output:
        return full_output.split(start, 1)[1].split(end, 1)[0].strip()
    return ""


def register_font_if_available(font_name: str) -> str:
    path = FONT_FILES.get(font_name)

    if font_name == "Helvetica" or not path:
        return "Helvetica"

    if os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
        except Exception:
            return "Helvetica"

    return "Helvetica"


def generate_pdf_bytes(text: str, font_name: str = "Helvetica", font_size: int = 11) -> bytes:
    """Gera um PDF simples com quebra de linha."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    font_to_use = register_font_if_available(font_name)
    c.setFont(font_to_use, font_size)

    left_margin = 2 * cm
    right_margin = 2 * cm
    top_margin = 2 * cm
    bottom_margin = 2 * cm
    max_width = width - left_margin - right_margin

    line_height = font_size * 1.4
    y = height - top_margin

    def wrap_line(line: str):
        if not line.strip():
            return [""]
        words = line.split()
        lines = []
        current = words[0]

        for word in words[1:]:
            test = current + " " + word
            if pdfmetrics.stringWidth(test, font_to_use, font_size) <= max_width:
                current = test
            else:
                lines.append(current)
                current = word

        lines.append(current)
        return lines

    for raw_line in text.splitlines():
        wrapped_lines = wrap_line(raw_line)

        for line in wrapped_lines:
            if y < bottom_margin:
                c.showPage()
                c.setFont(font_to_use, font_size)
                y = height - top_margin

            c.drawString(left_margin, y, line)
            y -= line_height

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_context_inputs(mode: str):
    """Renderiza os inputs do segundo bloco da interface."""
    is_grammar = (mode == "Análise gramatical e de clareza")

    st.subheader("2) Contexto")

    if is_grammar:
        notes = st.text_area(
            "Observações (opcional) — área/objetivo do currículo, tom desejado, partes que você quer revisar com mais atenção",
            height=220,
            key="notes",
        )
        return "", notes

    job_input_type = st.selectbox(
        "Formato da vaga:",
        ["Texto", "PDF", "Imagem (OCR)"],
        key="job_input_type",
    )

    if job_input_type == "Texto":
        job_text = st.text_area(
            "Cole aqui a descrição da vaga:",
            height=220,
            key="job_text"
        )
        return job_text, ""

    if job_input_type == "PDF":
        job_pdf = st.file_uploader(
            "Envie a vaga (PDF):",
            type=["pdf"],
            key="jobpdf"
        )
        job_text = ""
        if job_pdf:
            job_text = extract_text_from_pdf_bytes(job_pdf.read(), max_chars=120_000)
        return job_text, ""

    st.info("OCR não implementado neste MVP. Se a vaga vier como imagem, ainda será necessário adicionar OCR.")
    return "", ""


# ============================ Prompts ============================ #

def curriculum_analyser(llm, cv_content: str, notes: str = "") -> str:
    template = """
Atue como professora especialista em gramática normativa da língua portuguesa, com experiência na revisão de currículos profissionais.

Contexto adicional fornecido pelo usuário:
{notes}

Analise cuidadosamente o texto do currículo com base nos seguintes critérios:
- Coesão: conexão adequada entre frases, orações e parágrafos, fluidez textual e conectivos.
- Coerência: ideias organizadas de forma lógica, sem contradições ou ambiguidades.
- Formalidade: linguagem adequada ao contexto profissional, norma culta, sem gírias ou coloquialismos.
- Ortografia: erros de grafia.
- Gramática: concordância verbal/nominal, regência, pontuação, crase etc.

Para cada erro encontrado:
- Classifique a gravidade (leve, moderado ou grave).
- Apresente no formato:
  🔎 Trecho original
  ✅ Correção sugerida
  📘 Justificativa técnica

Caso o trecho esteja correto, confirme explicitamente que está adequado.

Ao final:
- Faça um resumo geral da qualidade textual do currículo.
- Dê uma nota de 0 a 10 considerando apenas os critérios linguísticos.
- Aponte sugestões estratégicas para tornar o texto mais claro, direto e profissional.

Seja técnica, objetiva e assertiva.

Currículo:
{cv}
"""
    prompt = template.format(cv=cv_content, notes=(notes or "Nenhum."))
    return llm_complete(llm, prompt)


def CVStrategicOptimizer(llm, cv_content: str, job_description: str) -> str:
    template = """
Atue como especialista em recrutamento, sistemas ATS (Applicant Tracking Systems) e redação estratégica de currículos.

Você receberá:
1) O currículo do candidato
2) A descrição de uma vaga específica

Sua tarefa é realizar uma análise técnica e estratégica seguindo as etapas abaixo:

ETAPA 1 — Extração de Palavras-Chave (da vaga)
Identifique palavras-chave técnicas, comportamentais e específicas do setor presentes na descrição da vaga.
Liste:
- Hard skills
- Soft skills
- Ferramentas
- Tecnologias
- Certificações
- Termos estratégicos repetidos

ETAPA 2 — Análise de Compatibilidade (CV x vaga)
Compare o currículo com a vaga e informe:
- Palavras-chave já presentes no currículo
- Palavras-chave ausentes
- Experiências que podem ser melhor descritas para alinhar com a vaga
- Nível estimado de compatibilidade (0% a 100%) com justificativa curta

ETAPA 3 — Otimização Estratégica para ATS (sem inventar)
Gere uma versão adaptada do currículo:
- Mantendo 100% da veracidade das informações
- Reorganizando e reescrevendo para incluir palavras-chave relevantes
- Melhorando clareza e objetividade
- Usando verbos de ação
- Priorizando termos compatíveis com ATS
- Adequando o resumo profissional para a empresa e vaga

ETAPA 4 — Ajuste Estratégico para a Empresa
Analise o tom da vaga e adapte o currículo para:
- Cultura mais técnica, corporativa ou inovadora
- Linguagem alinhada ao perfil da empresa
- Destaque das experiências e habilidades mais relevantes
- Condense o objetivo/resumo do candidato em uma frase estratégica, forte e profissional
- Seja conciso, sem perder impacto

ETAPA 5 — Entrega Estruturada
Responda exatamente neste formato:
1️⃣ Palavras-chave identificadas
2️⃣ Análise de compatibilidade
3️⃣ Sugestões estratégicas de melhoria
4️⃣ Currículo otimizado completo
5️⃣ Score estimado de otimização ATS (0–100)

IMPORTANTE:
No item 4️⃣, insira o currículo otimizado completo entre os marcadores abaixo, sem comentar sobre eles, sem explicá-los e sem citá-los fora do próprio bloco de conteúdo:
===CURRICULO_OTIMIZADO_INICIO===
(conteúdo do currículo otimizado)
===CURRICULO_OTIMIZADO_FIM===

Não invente informações que não estejam no currículo.

Currículo:
{cv}

Descrição da vaga:
{job}
"""
    prompt = template.format(cv=cv_content, job=job_description)
    return llm_complete(llm, prompt)


# ============================ UI Streamlit ============================ #

st.set_page_config(page_title="Análise de Currículos", page_icon="📄", layout="wide")
st.title("Análise e adequação de Currículos 📄")

with st.sidebar:
    st.subheader("Modelo (LlamaIndex)")
    model_choice = st.selectbox(
        "Escolha o modelo:",
        ["Gemini (Google)", "Groq (Llama 4 Maverick)"]
    )
    llm = get_llm(model_choice)

    limit = 200_000 if "Gemini" in model_choice else 40_000

    st.subheader("Currículo (PDF)")
    cv_file = st.file_uploader("Envie o currículo (PDF):", type=["pdf"])

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) O que você quer fazer?")
    mode = st.selectbox(
        "Selecione:",
        ["Otimização estratégica para vaga específica", "Análise gramatical e de clareza"],
        key="mode"
    )

with col2:
    job_text, notes = render_context_inputs(mode)

st.divider()

# ============================ Execução ============================ #

if cv_file:
    cv_content = extract_text_from_pdf_bytes(cv_file.read(), max_chars=limit)

    if not cv_content:
        st.error("Não consegui extrair texto do currículo. Se for PDF escaneado, precisará de OCR.")
        st.stop()

    with st.expander("Texto extraído do currículo (prévia)"):
        st.write(cv_content[:4000] + ("..." if len(cv_content) > 4000 else ""))

    is_grammar = (mode == "Análise gramatical e de clareza")

    if not is_grammar:
        if not job_text.strip():
            st.warning("Para otimização ATS, você precisa fornecer a descrição da vaga em texto ou PDF.")
            st.stop()

        with st.expander("Texto extraído da vaga (prévia)"):
            st.write(job_text[:4000] + ("..." if len(job_text) > 4000 else ""))

    if st.button("Executar análise", type="primary"):
        with st.spinner("Gerando resposta..."):
            if is_grammar:
                output = curriculum_analyser(llm, cv_content, notes=notes)
                st.session_state.optimized_cv = ""
            else:
                output = CVStrategicOptimizer(llm, cv_content, job_text)
                st.session_state.optimized_cv = extract_optimized_cv(output)

            st.session_state.last_output = output

        st.success("Concluído ✅")
        st.markdown(output)

    elif st.session_state.last_output:
        st.markdown(st.session_state.last_output)

    if st.session_state.optimized_cv:
        st.divider()
        st.subheader("Exportar currículo otimizado")

        font_choice = st.selectbox(
            "Escolha a fonte do PDF:",
            ["Helvetica", "Montserrat", "Merriweather", "Playfair Display", "Yanone Kaffeesatz", "Touvlo"],
            index=0,
            key="font_choice"
        )

        pdf_bytes = generate_pdf_bytes(
            st.session_state.optimized_cv,
            font_name=font_choice,
            font_size=11
        )

        st.download_button(
            label="⬇️ Baixar currículo otimizado em PDF",
            data=pdf_bytes,
            file_name="curriculo_otimizado.pdf",
            mime="application/pdf"
        )

else:
    st.info("Envie o currículo em PDF na barra lateral para começar.")
