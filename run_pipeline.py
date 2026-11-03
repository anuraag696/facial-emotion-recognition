# root/run_pipeline.py
from src.data.preprocess import preprocess_and_save
from src.models.train import train_model
from src.models.evaluate import evaluate_model

if __name__ == "__main__":
    print("========== FER PROJECT PIPELINE ==========")
    
    # Step 1: Preprocess raw images → processed
    print("\n[Step 1] Preprocessing images...")
    preprocess_and_save()
    
    # Step 2: Train the model
    print("\n[Step 2] Training the model...")
    train_model()
    
    # Step 3: Evaluate the model
    print("\n[Step 3] Evaluating the model...")
    evaluate_model()
    
    print("\nPipeline finished! Models saved in /models directory.")