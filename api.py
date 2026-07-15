from fastapi import FastAPI
from main import run_agent
from schemas import QuestionRequest, AnswerResponse

app = FastAPI()


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    # Run the agent with the provided question
    answer = run_agent(request.question, session_id=request.session_id)

    # Return the answer in the response model
    return AnswerResponse(answer=answer)
