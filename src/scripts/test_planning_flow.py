"""Test script for the multi-agent RAG flow with Planning Agent."""

import sys
import os
from pprint import pprint

# Add backend to sys.path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.core.agents.graph import run_qa_flow

def test_multi_topic_question():
    question = "What are the advantages of vector databases compared to traditional databases, and how do they handle scalability?"
    print(f"\n--- Testing Multi-Topic Question ---\nQuestion: {question}\n")
    
    try:
        final_state = run_qa_flow(question)
        
        print("\n--- Final Results ---")
        print(f"PLAN:\n{final_state.get('plan')}")
        print(f"\nSUB-QUESTIONS:\n{final_state.get('sub_questions')}")
        print(f"\nCONTEXT LENGTH: {len(final_state.get('context', ''))} characters")
        print(f"\nDRAFT ANSWER:\n{final_state.get('draft_answer')}")
        print(f"\nFINAL VERIFIED ANSWER:\n{final_state.get('answer')}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_topic_question()
