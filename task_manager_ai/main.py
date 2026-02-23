from fastapi import FastAPI
from pydantic import BaseModel
from agent_service import agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI() # קודם יוצרים את האפליקציה

# עכשיו מוסיפים את ההרשאות - זה פותר את שגיאת ה-405
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_agent(user_message: UserMessage):
    response = agent(user_message.message)
    return {"response": response}

@app.get("/")
async def read_root():
    return {"message": "Server is running!"}