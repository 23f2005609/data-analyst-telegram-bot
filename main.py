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
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "YOUR_AI_PIPE_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# ----------------- INITIALIZATION ----------------
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

client = OpenAI(
    api_key=AI_PIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI()

# Health check route for cron-job.org / Render
@app.get("/")
def health_check():
    return {"status": "Bot is awake and running!"}

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
    print(f"--- [MEMORY RESET] Chat ID: {chat_id} ---", flush=True)
    bot.reply_to(message, "Bot memory reset and ready for new tasks.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    chat_id = message.chat.id
    user_text = message.text
    print(f"\n--- [NEW TELEGRAM MESSAGE]: {user_text}", flush=True)
    
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
    
    max_turns = 10
    turn_count = 0
    empty_count = 0
    
    while turn_count < max_turns:
        turn_count += 1
        print(f"Calling LLM API (Turn {turn_count}/{max_turns})...", flush=True)
        try:
            response = client.chat.completions.create(
                model="openrouter/free", 
                messages=chat_histories[chat_id],
                tools=tools,
                temperature=0.0
            )
        except Exception as e:
            err_msg = f"API Error: {str(e)}"
            print(f"❌ {err_msg}", flush=True)
            bot.reply_to(message, err_msg)
            return
            
        msg = response.choices[0].message
        
        # Handle Tool Call execution
        if msg.tool_calls:
            print(f"🛠️ Model invoked {len(msg.tool_calls)} tool call(s)", flush=True)
            chat_histories[chat_id].append(msg.model_dump(exclude_unset=True))
            log_to_jsonl({"event": "tool_call", "chat_id": chat_id, "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "execute_python":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    print(f"🐍 Executing Code:\n{code}", flush=True)
                    
                    log_to_jsonl({"event": "execute_python_start", "code": code})
                    result = execute_python(code, chat_id)
                    print(f"📤 Code Output:\n{result}", flush=True)
                    log_to_jsonl({"event": "execute_python_result", "result": result})
                    
                    chat_histories[chat_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "execute_python",
                        "content": result
                    })
        # Handle Final Assistant Response
        else:
            raw_content = msg.content or ""
            final_text = raw_content.strip()
            print(f"💬 Model output text: {final_text[:100]}...", flush=True)

            if not final_text:
                empty_count += 1
                if empty_count >= 3:
                    bot.reply_to(message, "Model kept returning empty responses. Please send /start to try again.")
                    return
                print("⚠️ Empty content received, prompting model to proceed...", flush=True)
                chat_histories[chat_id].append({"role": "user", "content": "Please analyze the data and reply with the requested JSON object."})
                continue

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
                print("✅ Final JSON structured successfully. Sending reply to Telegram...", flush=True)
                bot.reply_to(message, json.dumps(response_json))
            except json.JSONDecodeError:
                print("⚠️ Output was not strict JSON. Sending raw response...", flush=True)
                bot.reply_to(message, final_text)
                
            break 

def run_telebot():
    print("🚀 Starting Telegram Bot polling thread...", flush=True)
    bot.infinity_polling()

@app.on_event("startup")
def on_startup():
    threading.Thread(target=run_telebot, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)