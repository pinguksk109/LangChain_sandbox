import ast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_google_community.search import GoogleSearchResults
from langchain_google_community.search import GoogleSearchAPIWrapper
from pydantic import BaseModel

llm = ChatGoogleGenerativeAI(model="gemini-pro")

class Answer(BaseModel):
    answer: str

prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="{user_input}について教えてください。"
)

api_wrapper = GoogleSearchAPIWrapper()
search_tool = GoogleSearchResults(api_wrapper=api_wrapper)
search_query = "日本一高い山"
search_results = search_tool.invoke(search_query)
search_results_list = ast.literal_eval(search_results)
snippets = [result['snippet'] for result in search_results_list]

structured_llm = llm.with_structured_output(Answer)

user_input = '日本一高い山'
context = "\n".join(snippets)
print(context)

prompt = prompt_template.format(user_input=user_input)

print('===')
print(prompt)
print('===')
result = structured_llm.invoke(prompt)

print(result)