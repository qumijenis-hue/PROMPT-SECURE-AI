import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import tiktoken
from transformers import pipeline

app = FastAPI(
    title="Prompt Analytics & Guardrails API",
    description="API to count tokens and measure text toxicity rates dynamically."
)

# 1. Initialize Tiktoken (cl100k_base is used by GPT-3.5 and GPT-4)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# 2. Initialize Toxicity Pipeline (Uses a lightweight BERT model optimized for toxicity)
# Note: The pipeline will automatically download the model on the first startup
try:
    TOXICITY_CLASSIFIER = pipeline(
        "text-classification", 
        model="unitary/toxic-bert", 
        top_k=None
    )
except Exception as e:
    print(f"Error loading toxicity model: {e}")
    TOXICITY_CLASSIFIER = None


# Input schema validation
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt text to inspect")


@app.post("/v1/analyze-prompt", status_code=status.HTTP_200_OK)
async def analyze_prompt(payload: PromptRequest):
    text = payload.prompt

    # ---- 1. Token Counting ----
    try:
        token_ids = TOKENIZER.encode(text)
        token_count = len(token_ids)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token counting error: {str(e)}"
        )

    # ---- 2. Toxicity Measurement ----
    toxicity_rate = 0.0
    detailed_scores = {}

    if TOXICITY_CLASSIFIER:
        try:
            # Run inference on the input text
            predictions = TOXICITY_CLASSIFIER(text)[0]
            
            # Map labels to their percentage scores
            detailed_scores = {pred['label']: round(pred['score'], 4) for pred in predictions}
            
            # Use the core 'toxic' category score as the primary toxicity rate
            toxicity_rate = detailed_scores.get('toxic', 0.0)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Toxicity analysis failed: {str(e)}"
            )
    else:
        detailed_scores = {"error": "Toxicity classifier model failed to load."}

    # ---- 3. Structural Decision/Flagging ----
    # Flag prompts that are highly toxic (e.g., > 70% probability)
    is_flagged = toxicity_rate > 0.70

    return {
        "analytics": {
            "token_count": token_count,
            "primary_toxicity_rate": toxicity_rate,
            "is_flagged": is_flagged
        },
        "detailed_metrics": detailed_scores,
        "input_preview": text[:100] + "..." if len(text) > 100 else text
    }


if __name__ == "__main__":
    import uvicorn
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)
