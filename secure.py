from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import tiktoken
from transformers import pipeline

app = FastAPI(
    title="Secure Prompt AI Gateway",
    description="API to batch-analyze prompts for toxicity and token metrics before LLM ingestion.",
    version="1.0.0"
)

# 1. Initialize tiktoken and Hugging Face pipelines globally
ENCODER = tiktoken.get_encoding("cl100k_base")

# Using a standard, fast pipeline for toxicity sequence classification
TOXICITY_CLASSIFIER = pipeline(
    "text-classification", 
    model="Xenova/toxic-bert", 
    top_k=None  # Returns scores for all categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)
)

# Configuration threshold
MAX_ALLOWED_TOKENS = 500

class PromptBatchRequest(BaseModel):
    prompts: list[str] = Field(..., min_items=1, description="List of input prompts to evaluate.")

class PromptAnalysisResult(BaseModel):
    prompt: str
    token_count: int
    is_flagged: bool
    flag_reason: str | None
    toxicity_scores: dict[str, float]

class BatchResponse(BaseModel):
    total_processed: int
    flagged_count: int
    results: list[PromptAnalysisResult]

@app.post(
    "/analyze-prompts", 
    response_model=BatchResponse, 
    status_code=status.HTTP_200_OK,
    summary="Analyze a batch of prompts for security and toxicity mitigation"
)
async def analyze_prompts(payload: PromptBatchRequest):
    input_prompts = payload.prompts
    
    # Remove empty or whitespace-only prompts
    cleaned_prompts = [p for p in input_prompts if p.strip()]
    if not cleaned_prompts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="The prompts list cannot be empty or contain only blank strings."
        )

    # 2. Batch inference on Hugging Face model for speed
    # This evaluates all prompts in parallel or optimal micro-batches internally
    try:
        hf_results = TOXICITY_CLASSIFIER(cleaned_prompts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hugging Face model inference failed: {str(e)}"
        )

    final_results = []
    flagged_count = 0

    # 3. Process metrics for each prompt
    for idx, prompt in enumerate(cleaned_prompts):
        # Calculate tokens using tiktoken
        token_count = len(ENCODER.encode(prompt))
        
        # Format toxicity classifications into a clean dictionary
        # hf_results[idx] looks like: [{'label': 'toxic', 'score': 0.9}, {'label': 'severe_toxic', 'score': 0.1}, ...]
        scores_dict = {item['label']: round(item['score'], 4) for item in hf_results[idx]}
        
        # Define rules for flagging a prompt
        is_flagged = False
        flag_reason = None
        
        # Check Rule A: Token Limit Breach
        if token_count > MAX_ALLOWED_TOKENS:
            is_flagged = True
            flag_reason = f"Token length ({token_count}) exceeds threshold of {MAX_ALLOWED_TOKENS}."
            
        # Check Rule B: Toxicity Threshold Breach (e.g., if generic toxicity score > 0.5)
        elif scores_dict.get("toxic", 0.0) > 0.5:
            is_flagged = True
            flag_reason = f"High toxicity rate detected ({scores_dict['toxic']})."
            
        # Check Rule C: Severe threats or hate speech breaches
        elif scores_dict.get("threat", 0.0) > 0.3 or scores_dict.get("identity_hate", 0.0) > 0.3:
            is_flagged = True
            flag_reason = "Critical safety violation (Threat/Hate speech category triggered)."

        if is_flagged:
            flagged_count += 1

        final_results.append(
            PromptAnalysisResult(
                prompt=prompt,
                token_count=token_count,
                is_flagged=is_flagged,
                flag_reason=flag_reason,
                toxicity_scores=scores_dict
            )
        )

    return BatchResponse(
        total_processed=len(cleaned_prompts),
        flagged_count=flagged_count,
        results=final_results
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("secure_api:app", host="127.0.0.1", port=8000, reload=True)
