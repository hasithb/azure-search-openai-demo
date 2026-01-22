import asyncio
import sys
import os

# Add app/backend to sys.path so we can import app
sys.path.append(os.path.join(os.getcwd(), "app", "backend"))

from app import create_app

async def run_test():
    print("Initializing app...")
    app = create_app()
    test_client = app.test_client()
    
    # 1. Simulate "Helpful" feedback (No context)
    payload_simple = {
        "message_id": "msg-local-simple",
        "rating": "helpful",
        "issues": [],
        "comment": "Quick thumbs up",
        "context_shared": False
    }
    
    # 2. Simulate "Unhelpful" feedback (With FULL context - mimicing UI)
    payload_rich = {
        "message_id": "msg-local-rich-context",
        "rating": "unhelpful",
        "issues": ["wrong_citation", "missing_info"],
        "comment": "The case law cited is from 1999, but it was overruled in 2005. Also missing the specific clause.",
        "context_shared": True,
        
        # UI Context Data
        "user_prompt": "What is the precedent for negligent misstatement?",
        "ai_response": "The leading case is Hedley Byrne v Heller [1964] AC 465. It established that a duty of care can arise...",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "How can I help with legal queries?"},
            {"role": "user", "content": "What is the precedent for negligent misstatement?"}
        ],
        "thoughts": [
            {
                "title": "Search Query",
                "description": "negligent misstatement precedent UK law"
            },
            {
                "title": "Retrieved Documents",
                "description": "[doc1] Hedley Byrne v Heller.pdf (Score: 0.95)\n[doc2] Caparo v Dickman.pdf (Score: 0.88)"
            }
        ]
    }
    
    print("\n--- Test 1: Simple Feedback ---")
    await test_client.post("/api/feedback", json=payload_simple)
    
    print("\n--- Test 2: Rich Context Feedback ---")
    print(f"Sending rich payload with {len(payload_rich['thoughts'])} thoughts and history...")
    response = await test_client.post("/api/feedback", json=payload_rich)
    
    print(f"Response status: {response.status_code}")
    if response.status_code == 200:
        data = await response.get_json()
        print(f"Response body: {data}")
        if data.get("status") == "received":
             print("SUCCESS: Rich feedback endpoint is working correctly.")
        else:
            print("FAILURE: Unexpected response body.")
    else:
        print("FAILURE: Endpoint returned non-200 status code.")

if __name__ == "__main__":
    asyncio.run(run_test())
