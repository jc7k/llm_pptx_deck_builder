# Create LangChain PRP

## Feature file: $ARGUMENTS

Generate a complete PRP for LangChain feature implementation with thorough research. Ensure context is passed to the AI agent to enable self-validation and iterative refinement. Read the feature file first to understand what needs to be created, how the examples provided help, and any other considerations.

The AI agent only gets the context you are appending to the PRP and training data. Assume the AI agent has access to the codebase and the same knowledge cutoff as you, so its important that your research findings are included or referenced in the PRP. The Agent has Websearch capabilities, so pass urls to documentation and examples.

## Research Process

1. **Codebase Analysis**
   - Search for similar features/patterns in the codebase
   - Identify files to reference in PRP
   - Note existing conventions to follow
   - Check test patterns for validation approach

2. **External Research**
   - Search for similar features/patterns online
   - Library documentation (include specific URLs)
   - Implementation examples (GitHub/StackOverflow/blogs)
   - Best practices and common pitfalls
   - Use Archon MCP server to gather latest LangChain documentation
   - Web search for LangGraph patterns and create_react_agent examples
   - Research LCEL (LangChain Expression Language) patterns
   - Investigate memory management and conversation persistence
   - Document streaming patterns and callback strategies   

3. **User Clarification** (if needed)
   - Specific patterns to mirror and where to find them?
   - Integration requirements and where to find them?

4. **Analyzing Initial Requirements**
   - Read and understand the agent feature requirements
   - Identify the type of agent needed (ReAct, conversational, LCEL chain, multi-agent)
   - Determine required model providers and external integrations
   - Assess complexity and scope of the agent implementation

5. **Agent Architecture Planning**
   - Design agent structure (agents/, chains/, tools/, memory/, prompts/)
   - Plan LCEL chain composition patterns
   - Design memory strategy (ConversationBufferMemory, ConversationSummaryMemory, etc.)
   - Plan tool registration with @tool decorator and structured inputs
   - Design testing approach with FakeLLM and mock patterns

6. **Implementation Blueprint Creation**
   - Create detailed agent implementation steps
   - Plan model provider configuration with ChatOpenAI, ChatAnthropic, etc.
   - Design tool error handling and retry mechanisms
   - Plan security implementation (API keys, input validation, rate limiting)
   - Design validation loops with agent behavior testing
   - Configure LangSmith tracing for observability

## PRP Generation

Using PRPs/templates/prp_langchain_base.md as template:

### Critical Context to Include and pass to the AI agent as part of the PRP
- **Documentation**: URLs with specific sections
  - https://python.langchain.com/docs/concepts/
  - https://langchain-ai.github.io/langgraph/
  - https://python.langchain.com/docs/concepts/lcel/
- **Code Examples**: Real snippets from codebase and LangChain docs
  - create_react_agent patterns
  - LCEL chain composition
  - Tool implementation with @tool
  - Memory management patterns
- **Gotchas**: Library quirks, streaming complexities, memory persistence
- **Patterns**: Existing LangChain approaches to follow

### Implementation Blueprint
- Start with pseudocode showing approach
- Reference real files for patterns
- Include error handling strategy
- List tasks to be completed to fulfill the PRP in the order they should be completed

### Validation Gates (Must be Executable) eg for python
```bash
# Syntax/Style
ruff check --fix && mypy .

# Unit Tests with FakeLLM
python -m pytest tests/test_agent.py -v

# Integration Tests
python -m pytest tests/ -v

# LangSmith Tracing Test
LANGCHAIN_TRACING_V2=true python tests/test_tracing.py
```

*** CRITICAL AFTER YOU ARE DONE RESEARCHING AND EXPLORING THE CODEBASE BEFORE YOU START WRITING THE PRP ***

*** ULTRATHINK ABOUT THE PRP AND PLAN YOUR APPROACH THEN START WRITING THE PRP ***

## Output
Save as: `PRPs/{feature-name}.md`

## Quality Checklist
- [ ] All necessary LangChain context included
- [ ] Validation gates are executable by AI
- [ ] References existing LangChain patterns
- [ ] Clear implementation path with LCEL/ReAct patterns
- [ ] Error handling and retry logic documented
- [ ] LangSmith tracing configuration included

Score the PRP on a scale of 1-10 (confidence level to succeed in one-pass implementation using claude codes)

Remember: The goal is one-pass implementation success through comprehensive context.