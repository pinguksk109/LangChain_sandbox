from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

llm = ChatGoogleGenerativeAI(model="gemini-pro")

prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="{user_input}について教えてください。"
)

class Answer(BaseModel):
    answer: str

structured_llm = llm.with_structured_output(Answer)

user_input = "日本一高い山"

prompt = prompt_template.format(user_input=user_input)

print(prompt)

result = structured_llm.invoke(prompt)

print(result)
