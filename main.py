import re,os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 读取.env文件
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek API端点
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek的模型名称
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

ask_fruit_unit_price:
e.g. ask_fruit_unit_price: apple
Asks the user for the price of a fruit

Example session:

Question: What is the unit price of apple?
Thought: I need to ask the user for the price of an apple to provide the unit price. 
Action: ask_fruit_unit_price: apple
PAUSE

You will be called again with this:

Observation: Apple unit price is 10/kg

You then output:

Answer: The unit price of apple is 10 per kg.
""".strip()


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def invoke(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=self.messages,
            temperature=0
        )
        return completion.choices[0].message.content


def calculate(what):
    return eval(what)

def ask_fruit_unit_price(fruit):
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)

action_re = re.compile(r'^Action: (\w+): (.*)$')

known_actions = {
    "calculate": calculate,
    "ask_fruit_unit_price": ask_fruit_unit_price
}

def query(question, max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = agent.invoke(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def calculate(what: str) -> float:
    """Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary"""
    return eval(what)

@tool
def ask_fruit_unit_price(fruit: str) -> str:
    """Asks the user for the price of a fruit"""
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)

prompt = PromptTemplate.from_template('''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}''')

tools = [calculate, ask_fruit_unit_price]
model = ChatOpenAI(model=DEFAULT_MODEL, api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "What is the total price of 3 kg of apple and 2 kg of banana?"
})
print(result)



