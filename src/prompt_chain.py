from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

llm_GP = ChatGoogleGenerativeAI(
    model="gemini-exp-1206",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

llm_GP_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
)

kategorizer_prompt = """
Erstelle zu folgendem Kontext folgende Datenpunkte in einem Json Format. Die Kategorien sind hier mit Key, Beschreibung und möglichen Werten aufgelistet:
- 'summary': Eine Zusammenfassung des Textes
- 'peer_review': Ist der Text Peer_Reviewed worden? (yes/no)
- 'about_parents': Ist der Text über Eltern? (yes/maybe/no)

---
Antworte im Json Format, wobei der Key dem Namen des PDFs entspricht und der Value ein Dictionary ist, mit den genannten Kategorien, jeweils dem von dir entsprechend gewählten Wert
---
Hier ist der Text und die Meta-Informationen des PDFs:
{context}
"""

kategorizer_template = PromptTemplate.from_template(kategorizer_prompt)
kategorizer_chain = (
    {
        "context": itemgetter("context"),
    }
    | kategorizer_template
    | llm_GP_flash
    | JsonOutputParser()
)
