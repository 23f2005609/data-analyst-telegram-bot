import os
import sys
import io
import json
import threading
import pandas as pd
import requests
import telebot
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from openai import OpenAI

# ----------------- CONFIGURATION -----------------
# 1. Swap OPENAI_API_KEY for your AI Pipe token
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "YOUR_AI_PIPE_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# ----------------- INITIALIZATION ----------------
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 2. Redirect the OpenAI client to AI Pipe's OpenRouter endpoint
client = OpenAI(
    api_key=AI_PIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI()

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/run.jsonl"
app.mount("/logs", StaticFiles(directory="logs"), name="logs")

chat_histories = {}
chat_envs = {}

def log_to_jsonl(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def execute_python(code: str, chat_id: int) -> str:
    if chat_id not in chat_envs:
        chat_envs[chat_id] = {"pd": pd, "requests": requests, "json": json, "os": os}
    
    env = chat_envs[chat_id]
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, env)
        output = redirected_output.getvalue()
        return output if output else "Code executed successfully with no output. (Tip: Use print() to see results)."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        sys.stdout = old_stdout

@bot.message_handler(commands=['start', 'reset'])
def handle_start(message):
    chat_id = message.chat.id
    chat_histories[chat_id] = []
    chat_envs[chat_id] = {}
    bot.reply_to(message, "Bot memory reset and ready for new tasks.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text
    
    if chat_id not in chat_histories or not chat_histories[chat_id]:
        chat_histories[chat_id] = [
            {
                "role": "system", 
                "content": (
                    "You are a Data Analyst Agent. The user will ask data-analysis questions, often referring to public datasets. "
                    "You have a tool 'execute_python' to run Python code. Use it to download data, read CSVs/Excel files, and perform calculations. "
                    "Your execution state is maintained between tool calls, so you can download data in one step and process it in the next. "
                    "Always use print() in your Python code to capture the output you want to see. "
                    "CRITICAL INSTRUCTION: When the user asks for a final JSON response, you MUST output ONLY ONE raw JSON object. "
                    "Do NOT wrap it in markdown blockquotes (e.g. ```json). "
                    "Include the exact 'answer' shape they requested, and include the 'log_url' key with the value 'LOG_URL_PLACEHOLDER'."
                )
            }
        ]
        chat_envs[chat_id] = {}
        
    chat_histories[chat_id].append({"role": "user", "content": user_text})
    log_to_jsonl({"event": "user_message", "chat_id": chat_id, "text": user_text})
    
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code to fetch and analyze data. You have access to pandas (pd) and requests. Always print() your final results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to run. Must use print() to output results."
                    }
                },
                "required": ["code"]
            }
        }
    }]
    
    while True:
        try:
            response = client.chat.completions.create(
                # 3. Use an OpenRouter formatted model name that supports tool calling
                model="openrouter/free", 
                messages=chat_histories[chat_id],
                tools=tools,
                temperature=0.0
            )
        except Exception as e:
            bot.reply_to(message, f"API Error: {str(e)}")
            return
            
        msg = response.choices[0].message
        
        if msg.tool_calls:
            chat_histories[chat_id].append(msg.model_dump(exclude_unset=True))
            log_to_jsonl({"event": "tool_call", "chat_id": chat_id, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "execute_python":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    
                    log_to_jsonl({"event": "execute_python_start", "code": code})
                    result = execute_python(code, chat_id)
                    log_to_jsonl({"event": "execute_python_result", "result": result})
                    
                    chat_histories[chat_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "execute_python",
                        "content": result
                    })
        else:
            # Safely handle cases where the LLM returns None for content
            raw_content = msg.content or ""
            final_text = raw_content.strip()

            if not final_text:
                bot.reply_to(message, "Received an empty response from the model. Please try again.")
                break

            chat_histories[chat_id].append({"role": "assistant", "content": final_text})
            log_to_jsonl({"event": "assistant_reply", "chat_id": chat_id, "text": final_text})
            
            clean_text = final_text
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "", 1)
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
            elif clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "", 1)
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
                
            try:
                response_json = json.loads(clean_text)
                response_json["log_url"] = f"{BASE_URL.rstrip('/')}/logs/run.jsonl"
                bot.reply_to(message, json.dumps(response_json))
            except json.JSONDecodeError:
                bot.reply_to(message, final_text)
                
            break 

def run_telebot():
    bot.infinity_polling()

@app.on_event("startup")
def on_startup():
    threading.Thread(target=run_telebot, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)