# Understanding Token Usage in RAG Chat

## How Prompts Are Created

The prompt generation process follows these steps:

1. **Search Query Generation**: The system first uses GPT to convert the user's question into an optimized search query
2. **Document Retrieval**: Azure AI Search retrieves relevant documents based on the query
3. **Prompt Assembly**: The retrieved documents are formatted and combined with the chat history and user question

## Token Usage Components

### 1. Input Tokens
- **Chat History**: Previous messages in the conversation
- **System Prompt**: Instructions for the AI model
- **Retrieved Documents**: Full content from search results
- **User Question**: The current query

### 2. Output Tokens
- **Generated Answer**: The AI's response
- **Citations**: References to source documents
- **Follow-up Questions**: Optional suggested questions

## Controlling Content Size

### Retrieve Count
The `retrieveCount` parameter (default: 10) controls how many documents are retrieved from the search index. Each document can contain:
- Full section content
- Metadata (sourcepage, sourcefile, category, etc.)
- Timestamps and URLs

### Token Limits by Model
- **GPT-3.5-turbo**: 4,096 tokens
- **GPT-3.5-turbo-16k**: 16,384 tokens
- **GPT-4**: 8,192 tokens
- **GPT-4-32k**: 32,768 tokens
- **GPT-4-turbo**: 16,384 tokens
- **GPT-4o/GPT-4o-mini**: 16,384 tokens

### Dynamic Token Allocation
The system now:
1. Calculates the total size of retrieved content
2. Estimates required tokens (≈4 characters per token)
3. Dynamically adjusts the response token limit
4. Ensures sufficient tokens for comprehensive answers

## Best Practices

1. **Increase Retrieve Count**: Set to 10-20 for comprehensive coverage
2. **Use Semantic Ranking**: Ensures most relevant content appears first
3. **Monitor Token Usage**: Check the "Thought Process" tab to see token allocation
4. **Adjust for Your Content**: 
   - Short documents: Higher retrieve count (15-20)
   - Long documents: Lower retrieve count (5-10)

## Viewing Token Usage

In the web interface:
1. Click the lightbulb icon on any answer
2. Select "Thought Process" tab
3. Look for "Token Usage Information" section

This shows:
- Total content size in characters
- Number of sources included
- Estimated prompt tokens
- Response token limit used
