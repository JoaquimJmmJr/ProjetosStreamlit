import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini

load_dotenv()


# ============================ Helpers ================================ #

def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = 200_000) -> str:
    """Extrai texto de PDF com PyMuPDF (bom para PDFs com texto selecionável)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    return text[:max_chars] if text else ""

def get_llm(model_choice: str):
    """Cria o LLM (LlamaIndex puro)."""
    if model_choice == "Gemini (Google)":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY não encontrado no .env")
            st.stop()
        # O Gemini no LlamaIndex usa a variável de ambiente, mas manter check é bom.
        return Gemini(model="models/gemini-2.5-flash")

    if model_choice == "Groq (Llama 4 Maverick)":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY não encontrado no .env")
            st.stop()
        return Groq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
        )

    st.error("Modelo inválido.")
    st.stop()

def llm_complete(llm, prompt: str) -> str:
    """Wrapper robusto para pegar o texto retornado pelo LlamaIndex."""
    resp = llm.complete(prompt)
    # Normalmente é resp.text
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    # fallback
    return str(resp).strip()

# ============================ Prompts ================================ #

def curriculum_analyser(llm, cv_content: str) -> str:
    template = """
Atue como professora especialista em gramática normativa da língua portuguesa, com experiência na revisão de currículos profissionais.

Analise cuidadosamente o texto do meu currículo com base nos seguintes critérios:
- Coesão: conexão adequada entre frases, orações e parágrafos, fluidez textual, conectivos.
- Coerência: ideias organizadas de forma lógica, sem contradições ou ambiguidades.
- Formalidade: linguagem adequada ao contexto profissional, norma culta, sem gírias/coloquialismos.
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
- Dê uma nota de 0 a 10 (considerando apenas critérios linguísticos).
- Aponte sugestões estratégicas para tornar o texto mais claro, direto e profissional.

Seja técnica, objetiva e assertiva.

Currículo:
{cv}
"""
    prompt = template.format(cv=cv_content)
    return llm_complete(llm, prompt)

def CVStrategicOptimizer(llm, cv_content: str, job_description: str) -> str:
    template = """
Atue como especialista em recrutamento, sistemas ATS (Applicant Tracking Systems) e redação estratégica de currículos.
Você receberá:
1) O currículo do candidato
2) A descrição de uma vaga específica

Sua tarefa é realizar uma análise técnica e estratégica seguindo as etapas abaixo:

ETAPA 1 — Extração de Palavras-Chave (da VAGA)
Identifique palavras-chave técnicas, comportamentais e específicas do setor presentes na descrição da vaga.
Liste:
- Hard skills
- Soft skills
- Ferramentas
- Tecnologias
- Certificações
- Termos estratégicos repetidos

ETAPA 2 — Análise de Compatibilidade (CV x VAGA)
Compare o currículo com a vaga e informe:
- Palavras-chave já presentes no currículo
- Palavras-chave ausentes
- Experiências que podem ser melhor descritas para alinhar com a vaga
- Nível estimado de compatibilidade (0% a 100%) com justificativa curta

ETAPA 3 — Otimização Estratégica para ATS (SEM INVENTAR)
Gere uma versão adaptada do currículo:
- Mantendo 100% da veracidade das informações (não inventar experiências/habilidades)
- Reorganizando e reescrevendo para incluir palavras-chave relevantes
- Melhorando clareza e objetividade
- Usando verbos de ação
- Priorizando termos compatíveis com ATS
- Adequando o resumo profissional para a empresa e vaga

ETAPA 4 — Ajuste Estratégico para a Empresa
Analise o tom da vaga e adapte o currículo para:
- Cultura mais técnica, corporativa ou inovadora
- Linguagem alinhada ao perfil da empresa
- Destaque de experiências e habilidades mais relevantes para o setor e tipo de empresa
- Condense as o objetivo/resumo do candidato em uma frase estratégica, forte e profissional, mantendo suas principais qualidades que podem contribuir para a vaga e a empresa
- Seja o mais conciso possível, sem perder impacto.  

ETAPA 5 — Entrega Estruturada
Responda exatamente neste formato:
1️⃣ Palavras-chave identificadas
2️⃣ Análise de compatibilidade
3️⃣ Sugestões estratégicas de melhoria
4️⃣ Versão otimizada completa do currículo
5️⃣ Score estimado de otimização ATS (0–100)

Seja técnico, estratégico e objetivo.
Não inclua elogios genéricos.
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
    model_choice = st.selectbox("Escolha o modelo:", ["Gemini (Google)", "Groq (Llama 4 Maverick)"])
    llm = get_llm(model_choice)

    # Limite por modelo (ajuste como quiser)
    limit = 200_000 if "Gemini" in model_choice else 40_000

    st.subheader("Currículo (PDF)")
    cv_file = st.file_uploader("Envie o currículo (PDF):", type=["pdf"])

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) O que você quer fazer?")
    mode = st.selectbox(
        "Selecione:",
        ["Otimização estratégica para vaga específica","Análise gramatical e de clareza"]
    )

# Variável que será usada para indicar que estamos no modo de análise gramatical para desabilitar campos
is_grammar = (mode == "Análise gramatical e de clareza")

with col2:
    st.subheader("2) Descrição da vaga")
    # Quando for análise gramatical, não pede "formato da vaga"
    job_input_type = st.selectbox(
        "Formato da vaga:",["Texto", "PDF", "Imagem (OCR)"],
        disabled=is_grammar,
        key="job_input_type")

# inicializando a variável que vai armazenar o texto da vaga
job_text = ""

if job_input_type == "Texto":
    job_text = st.text_area(
        "Cole aqui a descrição da vaga:", 
        height=220, 
        key="job_text",
        disabled=is_grammar
        )
elif job_input_type == "PDF":
    job_pdf = st.file_uploader(
        "Envie a vaga (PDF):", 
        type=["pdf"], 
        key="jobpdf", 
        disabled=is_grammar)
    if job_pdf:
        job_text = extract_text_from_pdf_bytes(job_pdf.read(), max_chars=120_000)
else:
    st.info("OCR não implementado neste MVP. Se você quiser, eu te passo a versão com pytesseract + dependências do SO.")
    job_text = ""

st.divider()

# ============================ Execução ============================ #

if cv_file:
    cv_content = extract_text_from_pdf_bytes(cv_file.read(), max_chars=limit)

    if not cv_content:
        st.error("Não consegui extrair texto do currículo. Se for PDF escaneado, precisará de OCR.")
        st.stop()

    with st.expander("Texto extraído do currículo (prévia)"):
        st.write(cv_content[:4000] + ("..." if len(cv_content) > 4000 else ""))

    if mode == "Otimização estratégica para vaga específica":
        if not job_text.strip():
            st.warning("Para otimização ATS, você precisa fornecer a descrição da vaga (texto ou PDF).")
            st.stop()

        with st.expander("Texto extraído da vaga (prévia)"):
            st.write(job_text[:4000] + ("..." if len(job_text) > 4000 else ""))

    if st.button("Executar análise", type="primary"):
        with st.spinner("Gerando resposta..."):
            if is_grammar:
                output = curriculum_analyser(llm, cv_content)
            else:
                output = CVStrategicOptimizer(llm, cv_content, job_text)

        st.success("Concluído ✅")
        st.markdown(output)

else:
    st.info("Envie o currículo em PDF na barra lateral para começar.")