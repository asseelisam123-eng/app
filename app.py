from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver


from dotenv import load_dotenv
from pathlib import Path
import os
from supabase import create_client


load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))





# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

import os

async def supabase_query_tool(args: dict):
    query = args.get("query", "").lower()

    # Competitions
    if any(word in query for word in ["competition", "event", "hackathon", "challenge"]):
        res = supabase.table("competitions").select("title_en, status, city").limit(5).execute()
        return res.data

    # FAQs
    elif any(word in query for word in ["faq", "question", "help"]):
        res = supabase.table("competition_faqs").select("question_en, answer_en").limit(5).execute()
        return res.data

    # Teams
    elif any(word in query for word in ["team", "group"]):
        res = supabase.table("teams").select("name, description, status").limit(5).execute()
        return res.data

    else:
        return [{"message": "No relevant data found"}]
    
tools = [
    {
        "name": "supabase_query",
        "description": "Fetch data from Supabase (competitions, teams, FAQs)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]



import os

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

llm_with_tools = llm.bind_tools(tools=tools)

async def model(state: State):
    system_prompt = """
You are an AI assistant connected to a Supabase database.

STRICT RULES:
- If the user asks about competitions → call supabase_query
- ALWAYS list ALL returned results clearly
- Do NOT summarize into one sentence
- Format as a list
"""

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]

    result = await llm_with_tools.ainvoke(messages)

    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
async def tool_node(state):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        if tool_call["name"] == "supabase_query":
            result = await supabase_query_tool(tool_call["args"])

            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name="supabase_query"
                )
            )

    return {"messages": tool_messages}

graph_builder = StateGraph(State)

graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")

graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")

graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

# Add CORS middleware with settings that match frontend requirements
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
    expose_headers=["Content-Type"], 
)

def serialise_ai_message_chunk(chunk): 
    if(isinstance(chunk, AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        # Generate new checkpoint ID for first message in conversation
        new_checkpoint_id = str(uuid4())

        config = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }
        
        # Initialize with first message
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )
        
        # First send the checkpoint ID
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        # Continue existing conversation
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            version="v2",
            config=config
        )

    async for event in events:
        event_type = event["event"]
        
        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            # Escape single quotes and newlines for safe JSON parsing
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
            
        elif event_type == "on_chat_model_end":
            # Check if there are tool calls for search
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            
           
    # Send an end event
    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )

from uuid import uuid4  # make sure this is at the top of your file if not already

@app.get("/chat")
async def chat(message: str):
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=message)]
        },
        config={
            "configurable": {
                "thread_id": str(uuid4())  
            }
        }
    )

    return {
        "response": result["messages"][-1].content
    }
# SSE - server-sent events 