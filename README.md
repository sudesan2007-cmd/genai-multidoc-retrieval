# Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex
## NAME: SUDESAN T
## REGISTER NUMBER: 212225240161



### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Manually reading and comparing multiple PDF documents is time-consuming. A RAG-based agent is needed to retrieve relevant information and answer questions across multiple documents efficiently.


### DESIGN STEPS:

#### STEP 1:
Load the three PDF documents and generate vector query and summary tools using LlamaIndex.


#### STEP 2:
Configure the GPT-4o-mini model and create an AI agent using FunctionCallingAgentWorker and AgentRunner.


#### STEP 3:
Submit a query to compare the evaluation datasets and results from all three PDFs and display the generated response.


### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/forum?id=BkxHEW5cF4",
    "https://openreview.net/forum?id=Knu0kEuJaR",
    "https://openreview.net/forum?id=IIMRxYrln8",
]

papers = [
    "Advantages and Challenges.pdf",
    "Data related.pdf",
    "End to end framework.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")

    safe_name = Path(paper).stem.replace(" ", "_")

    vector_tool, summary_tool = get_doc_tools(paper, safe_name)

    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

import llama_index.llms.openai.utils as openai_utils
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

openai_utils.ALL_AVAILABLE_MODELS["gpt-4o-mini"] = 128000
openai_utils.CHAT_MODELS["gpt-4o-mini"] = 128000

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini")

len(initial_tools)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)

agent = AgentRunner(agent_worker)

response = agent.query("Compare the evaluation datasets and results discussed in Advantages and Challenges.pdf, Data related.pdf, and End to end framework.pdf.")

print(response)

```





### OUTPUT:
<img width="1092" height="673" alt="Screenshot 2026-11-24 221531" src="https://github.com/user-attachments/assets/37e13066-4e46-4c49-8ccc-15f074958b1f" />



```


```


<img width="1073" height="712" alt="Screenshot 2026-11-24 221545" src="https://github.com/user-attachments/assets/587f0e6e-471a-41e0-b7e2-bde472c6de95" />





### RESULT:
The multi-document RAG agent retrieves relevant information from all three PDFs and generates a combined response comparing their evaluation datasets and results.

