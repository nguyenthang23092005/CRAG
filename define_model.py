from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
import os 
from dotenv import load_dotenv


load_dotenv("api.env")

llm_google = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.environ["GOOGLE_API_KEY"],
)

web_search = TavilySearch(k= 3, tavily_api_key = os.environ["TAVILY_API_KEY"])

