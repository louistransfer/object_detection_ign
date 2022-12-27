eval "$(conda shell.bash hook)"
conda activate satellite
uvicorn main:app --reload