from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
import os

KEY = os.environ.get("GOOGLE_API_KEY")
llm_GP = ChatGoogleGenerativeAI(
    model="gemini-exp-1206",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    api_key=KEY,
)

llm_GP_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    api_key=KEY,
)

kategorizer_prompt = """
Create the following data points for the given PDF context in a JSON format.  
The categories are listed below with a key, description, and possible values:

- 'summary': A summary of the text (string)

- 'peer_review': Was the study peer reviewed? (possible values: "yes" / "no")

- 'about_parents': Does the study focus on view of parents? (possible values: "yes" / "no")

- 'published_since_2014': Was the study published after the begining of 2014? (possible values: "yes" / "no")

- 'migration_immigration_asylum_refugee': Does the study address migration, immigration, asylum, or refugees? (possible values: "yes" / "no")

- 'transnational_parenthood_or_families': Is there a focus on transnational parenthood or transnational families? (possible values: "yes" / "no")

- 'research_type': What type of research was conducted? (possible values: "qualitative" / "quantitative" / "unclear")

---
Respond in JSON format, where the key is the first author of the PDF, and the value is a dictionary containing the categories listed above with the corresponding values determined by you.

Here is the text and meta-information of the PDF:
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
