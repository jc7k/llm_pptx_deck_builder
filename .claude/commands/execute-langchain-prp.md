# Execute LangChain Agent PRP

Implement a LangChain agent using the PRP file.

## PRP File: $ARGUMENTS

## Execution Process

1. **Load PRP**
   - Read the specified LangChain PRP file
   - Understand all agent requirements and research findings
   - Follow all instructions in the PRP and extend research if needed
   - Review LangChain documentation and examples for implementation guidance
   - Do more web searches and LangChain/LangGraph documentation review as needed

2. **ULTRATHINK**
   - Think hard before executing the agent implementation plan
   - Break down agent development into smaller steps using your todos tools  
   - Use the TodoWrite tool to create and track your agent implementation plan
   - Follow LangChain patterns for configuration and structure
   - Plan agents/, chains/, tools/, memory/, and testing approach

3. **Execute the plan**
   - Implement the LangChain agent following the PRP
   - Create agent with environment-based configuration (.env and config.py)
   - Use create_react_agent for tool-using agents or LCEL for chains
   - Implement tools with @tool decorators and proper error handling
   - Add comprehensive testing with FakeLLM and mock patterns

4. **Validate**
   - Test agent import and instantiation
   - Run FakeLLM validation for rapid development testing
   - Test tool registration and functionality
   - Run pytest test suite if created
   - Verify agent follows LangChain best practices
   - Test LangSmith tracing if configured

5. **Complete**
   - Ensure all PRP checklist items done
   - Test agent with example queries
   - Verify security patterns (environment variables, error handling)
   - Report completion status
   - Read the PRP again to ensure complete implementation

6. **Reference the PRP**
   - You can always reference the PRP again if needed

## LangChain-Specific Patterns to Follow

- **Configuration**: Use python-dotenv for environment variables  
- **Agents**: Use create_react_agent from LangGraph for tool-using agents
- **Chains**: Use LCEL (LangChain Expression Language) for composable chains
- **Tools**: Use @tool decorator with structured Pydantic inputs
- **Memory**: Use ConversationBufferMemory or ConversationSummaryMemory
- **Testing**: Include FakeLLM validation for development
- **Observability**: Enable LangSmith tracing with LANGCHAIN_TRACING_V2
- **Security**: Environment variables for API keys, proper error handling

Note: If validation fails, use error patterns in PRP to fix and retry. Follow LangChain documentation for proven implementation patterns.