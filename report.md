# 1. LangChain

>	LangChain is a framework designed to facilitate the development of applications powered by language models (LLMs) like OpenAI's GPT. It enables developers to integrate LLMs into their applications. 


  ![images](./image/langchain.png)

- **Prompt Templates**:Create and use templates for questions or commands to the language model.
- **Chains** : Combine multiple steps to handle complex tasks.
- **Agents**: Build automated applications that interact with the language model.
- **Memory**: Store and remember information from past action.
- **Retrieval Augmented Generation**: Combine the ability to retrieve information with the model’s text generation.

---
### Calling Prompts with LLMs: 
>  Combining LLMs with different components in a sequence. For instance, you might chain together an LLM to generate text, then send that text to an API, process the result, and send it back to the user.

![img](./image/lab1.webp)
-  Example code using Groq (Meta) (instead of using OpenAI, because their API key doesn't have a free version for development):
- install: `pip install langchain-groq langchain-core`
- ##### Code:
---
```py
from langchain_groq import ChatGroq
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
groq_api_key ='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'
chat = ChatGroq(
    api_key=groq_api_key,
    model_name = "llama3-8b-8192",
    temperature = 0
)
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system), ("human", human)
    ]
)
chain = prompt | chat | StrOutputParser()
r=  chain.invoke(
    {"text" : "What would a  good company name be for a company that makes colorful socks?"}
)
print(r)

```

- Analysis:
  - `from langchain_groq import ChatGroq` : This is the client for interacting with Groq's language model.    
  -  `from langchain_core.prompts import ChatPromptTemplate`: This is a component from LangChain that allows you to structure the input prompt.
  -  `from langchain_core.output_parsers import StrOutputParser`: This is used to parse the output into a string format.
  -  **API Key and Model Initialization**: 
     - `groq_api_key ='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'`: API key of Groq ( get from [Groq Key](https://console.groq.com/keys) ) 
     - `model_name = "llama3-8b-8192"`: specifies the language model.
     - `temperature = 0`: This parameter controls the randomness in the response (0 means the responses are more deterministic)
  - System and Human Prompts:
    - `system = "You are a helpful assistant."`: Defines the role of the system (the model will act as a helpful assistant).
    - `human = "{text}"`: This is a dynamic template, where {text} will be replaced by the user’s input.
  - Prompt Creation:
    ```python
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system), ("human", human)
    ]
    )
    ```
     This creates a prompt template using both system and human inputs

  - Chain Construction:
    - `prompt`: The structured prompt that includes both system and human inputs.
    - `chat`: The Groq LLM client (equivalent to the LLM model).
    - `StrOutputParser()`: Parses the output into a simple string.
  - Invocation and Output: The invoke method takes an input dictionary with text being the human query and processes it through the chain and after that print the answer to the screen
- After run this code: 
  ![images](./image/result1.png)

---
### Using different LLMs ( [Hugging Face Hub](https://huggingface.co/) )
> The Hugging Face Hub is the leading open platform for AI builders, with over 1 million models, datasets, and AI applications for building AI features that process and generate text, images, audio, video, and more.

 ![images](./image/lab2.png)

##### Code:
```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
from langchain import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0, "max_length": 64}
)

prompt = "What are good fitness tips?"

print(llm(prompt))
```
- `os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'`: Sets an environment variable HUGGINGFACEHUB_API_TOKEN with your Hugging Face API token from [API_TOKEN](https://huggingface.co/settings/tokens)
  

- Result:
![alt text](./image/image.png)

---
###  Prompt Templating and Chaining

> Prompt templating allows you to create dynamic, customizable prompts. It works by inserting specific values into a predefined text template, which is then passed to the language model. This provides flexibility and allows you to generate prompts that are tailored to specific tasks or queries.


##### Code:
```python
from langchain.chains.llm import LLMChain
from langchain_groq import ChatGroq
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
groq_api_key ='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'
chat = ChatGroq(
    api_key=groq_api_key,
    model_name = "llama3-8b-8192",
    temperature=0
)
system = "You are a helpful assistant."
human = "{text}"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system), ("human", human)
    ]
)
chain = prompt | chat | StrOutputParser()
template = "What is the naming consultant for new companies. What is a good name for a company that makes {product} ?"
format_prompt = PromptTemplate.from_template(template)
data=  format_prompt.format(product="colorful socks")
first_chain = LLMChain(prompt=format_prompt, llm=chat)
print(first_chain.run("colorful socks"))

```
 - What is temple?
```python
    template = "What is the naming consultant for new companies. What is a good name for a company that makes {product} ?"
    format_prompt = PromptTemplate.from_template(template)

```
> Here, `format_prompt` is a template with a placeholder `{product} `that can be replaced dynamically with different product names.


- For example with `{product}` = `colorful socks` : 
  ![alt text](./image/lab4.png)  

---
### Using Simple Sequential Chains 
> Simple Sequential Chains in LangChain allow you to chain together multiple language model chains, where the output of one chain becomes the input for the next. This is useful when you want to perform a multi-step task, where each step builds upon the previous one.

![image](./image/img5.png)

##### Code: 
```python
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_groq import ChatGroq
from langchain_core.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from sympy.physics.units import temperature
groq_api_key ='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'
chat = ChatGroq(
    api_key=groq_api_key,
    model_name = "llama3-8b-8192",
    temperature=0
)
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system), ("human", human)
    ]
)
chain = prompt | chat | StrOutputParser()
first_template = "What is the naming consultant for new companies. What is a good name for a company that makes {product} ?"
first_prompt = PromptTemplate.from_template(first_template)
first_chain = LLMChain(prompt=first_prompt, llm=chat)
second_template = "Write a catch phrase for the following company : {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(prompt=second_prompt, llm=chat)
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)

```

- Here, `first_template` is a prompt template where `{product}` will be replaced dynamically with the `product` .
- The first chain `first_chain` that takes a product name as input and use to generate a company name.
- The `second_template` takes a company name as input and asks the model to generate a catchphrase.
- `second_chain` takes the output from  `first_chain` `(company name)` as input and generates a catchphrase.
- SimpleSequentialChain chains together `first_chain` and `second_chain`.
- The output of `first_chain` `(company name)` is automatically passed as input to `second_chain` (to generate a catchphrase).
- And example run with value  = `colorful socks` : 

![alt text](./image/lab5-1.png)
![alt text](./image/lab5-2.png)


--- 

### Action Agents 
> An agent uses an LLM to choose a sequence of actions to take to solve a problem. In chains, a sequence of actions is hardcoded. In agents, an LLM uses its reasoning to choose which actions to take and in which order

![img](./image/action.jpg)

##### Code: 
```python

from langchain.agents import load_tools,initialize_agent

from langchain_groq import ChatGroq
from langchain_core.prompts import  ChatPromptTemplate

groq_api_key ='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'
llm = ChatGroq(
    api_key=groq_api_key,
    model_name = "llama3-8b-8192",
    temperature=0
)
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system), ("human", human)
    ]
)
prompt = "When was the 3rd president of the US born? What is that years raised to the power of 3?"

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(tools,llm, agent ="zero-shot-react-description", verbose=True)
agent.run(prompt)
```
- Load Tools:`load_tools` loads specific tools that the agent can use. In this case, `"wikipedia"`  and `"llm-math"` are tools that help with information and mathematical. 

- Initialize Agent:`initialize_agent` sets up the agent with the loaded tools and language model. The `"zero-shot-react-description"` parameter specifies the type of agent to initialize. The `verbose=True` argument controls the level of logging output.
- Result after running with data from Wikipedia (some results are wrong due to calculation errors): 
![alt text](./image/lab6.png)

--- 

### Overall and purpose

> LangChain is a powerful tool for building chatbots. With its features like Prompt Templates, Chains, Agents, Memory, and Retrieval Augmented Generation (RAG), it helps create chatbots that are smart and responsive. These tools make it easier to design chatbot interactions, manage conversations, and remember past interactions, which leads to a better user experience. By using LangChain for your website’s chatbot, you can make it more engaging and effective. Looking ahead, there’s a lot of potential to explore how LangChain can be used even more to improve chatbots and other interactive applications. 