#!/usr/bin/env python3
"""Quick test of evaluation components."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_direct_evaluation import (
    get_search_client,
    get_openai_client,
    search_documents,
    generate_rag_response,
)

async def test():
    print("=" * 80)
    print("TESTING V2 INDEX EVALUATION COMPONENTS")
    print("=" * 80)
    
    # Test search
    print("\n1. Testing Search Client...")
    search_client, search_cred = await get_search_client()
    print("✓ Search client created for index: legal-court-rag-index-v2")
    
    results = await search_documents(search_client, 'CPR Part 35 experts', top=3)
    print(f"✓ Search results: {len(results)} documents found")
    if results:
        print(f"  First result: {results[0]['sourcepage']}")
        print(f"  Content preview: {results[0]['content'][:100]}...")
    
    # Test OpenAI
    print("\n2. Testing OpenAI Client...")
    openai_client, deployment, openai_cred = await get_openai_client()
    print(f"✓ OpenAI client created with deployment: {deployment}")
    
    # Test RAG response generation
    print("\n3. Testing RAG Response Generation...")
    if results:
        try:
            response = await generate_rag_response(
                openai_client,
                deployment,
                'What are the duties of expert witnesses under CPR Part 35?',
                results
            )
            print(f"✓ Response generated: {len(response)} characters")
            print(f"\nResponse preview:")
            print("-" * 80)
            print(response[:500] if len(response) > 500 else response)
            print("-" * 80)
        except Exception as e:
            print(f"✗ Error generating response: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    await search_client.close()
    await openai_client.close()
    await search_cred.close()
    await openai_cred.close()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test())
