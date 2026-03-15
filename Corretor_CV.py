import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def split_optimizer_sections(text: str):
    sections = {
        "keywords": "",
        "compatibility": "",
        "suggestions": "",
        "optimized": "",
        "justification": ""
    }

    try:
        after_1 = text.split("1️⃣", 1)[1]
        part_1, rest = after_1.split("2️⃣", 1)
        sections["keywords"] = part_1.strip()

        part_2, rest = rest.split("3️⃣", 1)
        sections["compatibility"] = part_2.strip()

        part_3, rest = rest.split("4️⃣", 1)
        sections["suggestions"] = part_3.strip()

        if "5️⃣" in rest:
            part_4, part_5 = rest.split("5️⃣", 1)
            sections["optimized"] = part_4.strip()
            sections["justification"] = part_5.strip()
        else:
            sections["optimized"] = rest.strip()

    except Exception:
        sections["optimized"] = text.strip()

    return sections

def format_keywords_section(text: str) -> str:
    """
    Tenta transformar a seção de palavras-chave em bullets legíveis.
    """
    labels = [
        "Hard Skills",
        "Soft Skills",
        "Ferramentas",
        "Tecnologias",
        "Certificações",
        "Termos Estratégicos Repetidos",
        "Termos Estratégicos"
    ]

    formatted_lines = []
    working_text = text.replace("\n", " ")

    for label in labels:
        pattern = rf"{label}\s*:\s*(.*?)(?=(Hard Skills|Soft Skills|Ferramentas|Tecnologias|Certificações|Termos Estratégicos Repetidos|Termos Estratégicos)\s*:|$)"
        match = re.search(pattern, working_text, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip(" .")
            formatted_lines.append(f"- {label}: {content}")

    if formatted_lines:
        return "\n".join(formatted_lines)

    return text

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def split_items_from_line(line: str) -> list[str]:
    """
    Converte uma linha como:
    '- Hard Skills: Python, C++, Arduino'
    em ['python', 'c++', 'arduino']
    """
    if ":" in line:
        content = line.split(":", 1)[1]
    else:
        content = line

    parts = re.split(r"[,\n;•\-]+", content)
    return [p.strip().lower() for p in parts if p.strip()]

def extract_keywords_dict(text: str) -> dict:
    """
    Extrai as categorias da seção 1️⃣ Palavras-chave identificadas.
    """
    labels_map = {
        "hard_skills": ["hard skills", "hard skills"],
        "soft_skills": ["soft skills"],
        "ferramentas": ["ferramentas"],
        "tecnologias": ["tecnologias"],
        "certificacoes": ["certificações", "certificacoes"],
        "termos_estrategicos": ["termos estratégicos repetidos", "termos estratégicos", "termos estrategicos repetidos", "termos estrategicos"],
    }

    result = {k: [] for k in labels_map.keys()}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        clean_line = line.lstrip("-• ").strip()
        lower_line = clean_line.lower()

        for key, aliases in labels_map.items():
            if any(lower_line.startswith(alias) for alias in aliases):
                result[key] = split_items_from_line(clean_line)

    return result

def keyword_match_score(cv_text: str, keywords_dict: dict) -> tuple[int, dict]:
    """
    Calcula presença literal de palavras-chave no currículo.
    """
    cv_norm = normalize_text(cv_text)

    weighted_groups = {
        "hard_skills": 0.35,
        "tecnologias": 0.25,
        "ferramentas": 0.15,
        "certificacoes": 0.10,
        "soft_skills": 0.10,
        "termos_estrategicos": 0.05,
    }

    group_details = {}
    weighted_score = 0.0

    for group, weight in weighted_groups.items():
        items = keywords_dict.get(group, [])
        if not items:
            group_details[group] = {"score": 0, "matched": [], "missing": []}
            continue

        matched = []
        missing = []

        for item in items:
            if item and item in cv_norm:
                matched.append(item)
            else:
                missing.append(item)

        group_score = len(matched) / max(len(items), 1)
        weighted_score += group_score * weight

        group_details[group] = {
            "score": round(group_score * 100, 1),
            "matched": matched,
            "missing": missing,
        }

    return round(weighted_score * 100), group_details

def semantic_similarity_score(cv_text: str, job_text: str) -> int:
    """
    Similaridade semântica global entre currículo e vaga.
    """
    model = get_embedding_model()

    embeddings = model.encode([cv_text[:8000], job_text[:8000]])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # converte -1..1 para 0..100
    score = max(0.0, min(float(sim), 1.0)) * 100
    return round(score)

def sentence_level_alignment_score(cv_text: str, job_text: str) -> int:
    """
    Mede alinhamento entre frases/requisitos da vaga e trechos do CV.
    """
    model = get_embedding_model()

    cv_sentences = [s.strip() for s in re.split(r"[.\n]+", cv_text) if len(s.strip()) > 20]
    job_sentences = [s.strip() for s in re.split(r"[.\n]+", job_text) if len(s.strip()) > 20]

    cv_sentences = cv_sentences[:40]
    job_sentences = job_sentences[:25]

    if not cv_sentences or not job_sentences:
        return 0

    cv_emb = model.encode(cv_sentences)
    job_emb = model.encode(job_sentences)

    sims = cosine_similarity(job_emb, cv_emb)
    best_matches = sims.max(axis=1)

    return round(float(np.mean(best_matches)) * 100)

def section_presence_score(cv_text: str) -> tuple[int, dict]:
    """
    Verifica se o currículo tem seções ATS-friendly.
    """
    cv_norm = normalize_text(cv_text)

    patterns = {
        "resumo": ["resumo", "objetivo", "summary", "professional summary"],
        "formacao": ["formação", "educação", "education", "academic background"],
        "competencias": ["competências", "habilidades", "skills", "technical skills"],
        "projetos_experiencias": ["projetos", "experiências", "experience", "projects"],
        "certificacoes": ["certificações", "certifications"],
        "idiomas": ["idiomas", "languages"],
    }

    found = {}
    hits = 0

    for section, aliases in patterns.items():
        present = any(alias in cv_norm for alias in aliases)
        found[section] = present
        if present:
            hits += 1

    score = round((hits / len(patterns)) * 100)
    return score, found

def compute_hybrid_ats_score(cv_text: str, job_text: str, keywords_section_text: str) -> dict:
    """
    Score final híbrido:
    - 35% keywords
    - 30% similaridade semântica global
    - 20% alinhamento frase a frase
    - 15% presença de seções
    """
    keywords_dict = extract_keywords_dict(keywords_section_text)

    keyword_score, keyword_details = keyword_match_score(cv_text, keywords_dict)
    semantic_score = semantic_similarity_score(cv_text, job_text)
    alignment_score = sentence_level_alignment_score(cv_text, job_text)
    sections_score, sections_found = section_presence_score(cv_text)

    final_score = round(
        keyword_score * 0.35
        + semantic_score * 0.30
        + alignment_score * 0.20
        + sections_score * 0.15
    )

    return {
        "final_score": final_score,
        "keyword_score": keyword_score,
        "semantic_score": semantic_score,
        "alignment_score": alignment_score,
        "sections_score": sections_score,
        "keyword_details": keyword_details,
        "sections_found": sections_found,
    }

def build_ats_justification(score_data: dict) -> str:
    lines = [
        f"**Score por palavras-chave:** {score_data['keyword_score']}/100",
        f"**Similaridade semântica global:** {score_data['semantic_score']}/100",
        f"**Alinhamento entre requisitos e experiências:** {score_data['alignment_score']}/100",
        f"**Presença de seções ATS-friendly:** {score_data['sections_score']}/100",
        "",
        "**Seções identificadas no currículo:**"
    ]

    section_labels = {
        "resumo": "Resumo/Objetivo",
        "formacao": "Formação",
        "competencias": "Competências/Habilidades",
        "projetos_experiencias": "Projetos/Experiências",
        "certificacoes": "Certificações",
        "idiomas": "Idiomas",
    }

    for key, label in section_labels.items():
        status = "✅" if score_data["sections_found"].get(key) else "❌"
        lines.append(f"- {status} {label}")

    lines.append("")
    lines.append("**Resumo de match por palavras-chave:**")

    group_labels = {
        "hard_skills": "Hard Skills",
        "tecnologias": "Tecnologias",
        "ferramentas": "Ferramentas",
        "certificacoes": "Certificações",
        "soft_skills": "Soft Skills",
        "termos_estrategicos": "Termos Estratégicos",
    }

    for group, label in group_labels.items():
        group_data = score_data["keyword_details"].get(group, {})
        matched = ", ".join(group_data.get("matched", [])[:8]) or "nenhuma"
        missing = ", ".join(group_data.get("missing", [])[:8]) or "nenhuma"
        lines.append(f"- **{label}** — Match: {group_data.get('score', 0)}/100")
        lines.append(f"  - Encontradas: {matched}")
        lines.append(f"  - Ausentes: {missing}")

    return "\n".join(lines)

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
Liste e apresente a seção obrigatoriamente em formato de lista com bullets, uma categoria por linha, exatamente neste estilo:
**Hard skills**: ...
**Soft skills**: ...
**Ferramentas**: ...
**Tecnologias**: ...
**Certificações**: ...
**Termos estratégicos repetidos**: ...
Não escreva essa seção em parágrafo corrido.

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
4️⃣ Versão otimizada completa do currículo (entregue sem comentários, apenas o texto otimizado)
5️⃣ Justificativa final de aderência à vaga (não forneça nota numérica. Explique de forma objetiva os principais pontos fortes do currículo em relação à vaga e os principais gaps restantes)

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

def CVEnglishVersionGenerator(llm, cv_content: str) -> str:
    template = """
    You are a professional resume editor and English language specialist with experience in international recruiting and ATS systems.
    Your task is to convert the following resume into a **fully professional English version** suitable for international companies.
    Follow these rules carefully:

    1. Translate the entire resume into **natural, fluent, and professional English**.
    2. Preserve all factual information. **Do not invent or remove experiences.**
    3. Improve clarity, coherence, and conciseness where necessary.
    4. Use vocabulary commonly found in **international technical resumes**.
    5. Ensure the text is grammatically correct and stylistically professional.
    6. Adapt section titles to standard English resume sections, such as:
    - Professional Summary
    - Education
    - Technical Skills
    - Projects
    - Certifications
    - Languages
    7. Rewrite sentences when needed to sound **natural in English**, not like a literal translation.
    8. Use action verbs commonly used in professional resumes.
    9. Maintain a clean and structured resume format.
    10. Use concise bullet points and strong action verbs commonly used in professional resumes such as: Developed, Implemented, Designed, Built, Led, Integrated, Optimized, Automated, Analyzed.

    Important constraints:
    - Do not include explanations.
    - Do not include translation notes.
    - Output **only the final English resume**.
    - The resume must read as if it were originally written in English.

    Resume:
    {cv}
    """
    prompt = template.format(cv=cv_content)
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
        ["Otimização estratégica para vaga específica","Generate English Resume Version","Análise gramatical e de clareza"]
    )

# Variável que será usada para indicar que estamos no modo de análise gramatical para desabilitar campos
is_grammar = (mode == "Análise gramatical e de clareza" or mode == "Generate English Resume Version")

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
            if mode == "Otimização estratégica para vaga específica":
                output = CVStrategicOptimizer(llm, cv_content, job_text)
            elif mode == "Generate English Resume Version":
                 output = CVEnglishVersionGenerator(llm, cv_content)
            else:
                # mode == "Análise gramatical e de clareza"
                output = curriculum_analyser(llm, cv_content)

        st.success("Concluído ✅")
        if not is_grammar:
            sections = split_optimizer_sections(output)

            tab1, tab2, tab3, tab4 = st.tabs([
                "1️⃣ Palavras-chave identificadas",
                "2️⃣ Análise de compatibilidade",
                "3️⃣ Sugestões estratégicas",
                "4️⃣ Versão otimizada + Score ATS"
            ])

            with tab1:
                st.markdown(format_keywords_section(sections["keywords"]))
            with tab2:
                st.markdown(sections["compatibility"])
            with tab3:
                st.markdown(sections["suggestions"])
            with tab4:
                st.markdown("### Currículo otimizado")
                st.markdown(sections["optimized"])

                hybrid_score = compute_hybrid_ats_score(
                    cv_text=cv_content,
                    job_text=job_text,
                    keywords_section_text=sections["keywords"]
                )

                st.divider()
                st.metric("ATS Score", f"{hybrid_score['final_score']}/100")
                st.progress(hybrid_score["final_score"] / 100)

                st.markdown("### Justificativa do score")
                st.markdown(build_ats_justification(hybrid_score))

                if sections["justification"]:
                    st.markdown("### Justificativa adicional do modelo")
                    st.markdown(sections["justification"])

        else:
            st.markdown(output)

else:
    st.info("Envie o currículo em PDF na barra lateral para começar.")
