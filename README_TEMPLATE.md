# LangChain Context Engineering Template

A comprehensive template for building production-grade AI agents using LangChain with context engineering best practices, LCEL chains, ReAct agents, memory management, and comprehensive testing patterns.

## 🚀 Quick Start - Copy Template

**Get started in 2 minutes:**

```bash
# Clone the context engineering repository
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro/use-cases/langchain

# 1. Copy this template to your new project
python copy_template.py /path/to/my-agent-project

# 2. Navigate to your project
cd /path/to/my-agent-project

# 3. Start building with the PRP workflow
# Fill out PRPs/INITIAL.md with the agent you want to create

# 4. Generate the PRP based on your detailed requirements (validate the PRP after generating!)
/generate-langchain-prp PRPs/INITIAL.md

# 5. Execute the PRP to create your LangChain agent
/execute-langchain-prp PRPs/generated_prp.md
```

If you are not using Claude Code, you can simply tell your AI coding assistant to use the generate-langchain-prp and execute-langchain-prp slash commands in .claude/commands as prompts.

## 📖 What is This Template?

This template provides everything you need to build sophisticated LangChain agents using proven context engineering workflows. It combines:

- **LangChain Best Practices**: ReAct agents, LCEL chains, memory management, and tool integration
- **Context Engineering Workflows**: Proven PRP (Product Requirements Prompts) methodology
- **Working Examples**: Complete agent implementations you can learn from and extend

## 🎯 PRP Framework Workflow

This template uses a 3-step context engineering workflow for building AI agents:

### 1. **Define Requirements** (`PRPs/INITIAL.md`)
Start by clearly defining what your agent needs to do:
```markdown
# Customer Support Agent - Initial Requirements

## Overview
Build an intelligent customer support agent that can handle inquiries, 
access customer data, and escalate issues appropriately.

## Core Requirements
- Multi-turn conversations with memory persistence
- Customer authentication and account access
- Account balance and transaction queries
- Payment processing and refund handling
- Tool integration for external systems
...
```

### 2. **Generate Implementation Plan** 
```bash
/generate-langchain-prp PRPs/INITIAL.md
```
This creates a comprehensive 'Product Requirements Prompts' document that includes:
- LangChain technology research and best practices
- Agent architecture design with ReAct patterns or LCEL chains
- Implementation roadmap with validation loops
- Memory management and conversation persistence
- Security patterns and production considerations

### 3. **Execute Implementation**
```bash
/execute-langchain-prp PRPs/your_agent.md
```
This implements the complete agent based on the PRP, including:
- Agent creation with create_react_agent or LCEL chains
- Tool integration with @tool decorators and error handling
- Memory management with ConversationBufferMemory or ConversationSummaryMemory
- Comprehensive testing with FakeLLM and mock patterns
- LangSmith tracing for observability

## 📂 Template Structure

```
langchain/
├── CLAUDE.md                           # LangChain global development rules
├── copy_template.py                    # Template deployment script
├── .claude/commands/
│   ├── generate-langchain-prp.md       # PRP generation for agents
│   └── execute-langchain-prp.md        # PRP execution for agents
├── PRPs/
│   ├── templates/
│   │   └── prp_langchain_base.md       # Base PRP template for agents
│   └── INITIAL.md                      # Example agent requirements
├── examples/
│   ├── react_agent/                    # ReAct agent with tools
│   │   ├── agent.py                    # Agent with web search and calculator
│   │   └── README.md                   # Usage guide
│   ├── conversational_agent/           # Agent with memory
│   │   ├── agent.py                    # Conversation management
│   │   └── requirements.txt            # Dependencies
│   ├── lcel_chains/                    # LCEL chain examples
│   │   ├── simple_chain.py             # Basic prompt -> LLM -> parser
│   │   └── complex_chain.py            # Multi-step reasoning chain
│   └── testing_examples/               # Comprehensive testing patterns
│       ├── test_agent_patterns.py      # FakeLLM, mock examples
│       └── pytest.ini                  # Test configuration
└── README.md                           # This file
```

## 🤖 Agent Examples Included

### 1. ReAct Agent (`examples/react_agent/`)
**Tool-using agent with reasoning capabilities**:
- Environment-based configuration with python-dotenv
- create_react_agent from LangGraph
- Tool integration with external APIs (web search, calculator)
- Proper error handling and retry mechanisms

**Key Files:**
- `agent.py`: ReAct agent with tool calling capabilities
- `tools.py`: Custom tools with @tool decorator
- `config.py`: Environment configuration management
- `memory.py`: Conversation memory management

### 2. Conversational Agent (`examples/conversational_agent/`)
A memory-enabled conversational agent demonstrating core patterns:
- **Memory management** with ConversationBufferMemory
- **Environment-based model configuration**
- System prompts with ChatPromptTemplate
- Conversation context tracking and persistence

**Key Features:**
- Simple conversational interface
- Memory-based context management
- Settings-based configuration pattern
- Clean, minimal implementation

### 3. LCEL Chains (`examples/lcel_chains/`)
Composable chains using LangChain Expression Language:
- **Simple chains** with prompt -> LLM -> parser
- **Complex chains** with multi-step reasoning
- RunnablePassthrough and RunnableLambda patterns
- Chain composition and branching

**Key Features:**
- LCEL (LangChain Expression Language) patterns
- Composable and reusable chain components
- Input/output transformation patterns
- Error handling in chain execution

### 4. Multi-Agent System (`examples/multi_agent/`)
**NEW**: Shows agent orchestration patterns:
- **Agent supervisor** with task routing
- **Specialized agents** for different domains
- **State management** across agent interactions
- **Handoff patterns** between agents

**Key Features:**
- Demonstrates agent composition patterns
- Task routing and delegation
- State sharing between agents
- Clear documentation on multi-agent architecture

### 5. Testing Examples (`examples/testing_examples/`)
Comprehensive testing patterns for LangChain agents:
- FakeLLM for deterministic testing
- Mock tool responses and external services
- Integration testing with real models
- LangSmith tracing validation

**Key Features:**
- Unit testing without API costs
- Mock dependency injection
- Tool validation and error scenario testing
- Integration testing patterns with pytest

## 🔧 Key LangChain Components

### ReAct Agents
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_react_agent(
    model=model,
    tools=[web_search, calculator],
    prompt=system_prompt
)
```

### LCEL Chains
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Tell me about {topic}")
    | ChatOpenAI(model="gpt-4")
    | StrOutputParser()
)
```

### Memory Management
```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### Tool Integration
```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return results
```

## 📚 Additional Resources

- **Official LangChain Documentation**: https://python.langchain.com/docs/
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LCEL Guide**: https://python.langchain.com/docs/concepts/lcel/
- **LangSmith Observability**: https://smith.langchain.com/
- **Context Engineering Methodology**: See main repository README

## 🆘 Support & Contributing

- **Issues**: Report problems with the template or examples
- **Improvements**: Contribute additional examples or patterns
- **Questions**: Ask about LangChain integration or context engineering

This template is part of the larger Context Engineering framework. See the main repository for more context engineering templates and methodologies.

---

**Ready to build production-grade LangChain agents?** Start with `python copy_template.py my-agent-project` and follow the PRP workflow! 🚀