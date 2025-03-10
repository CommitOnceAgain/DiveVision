# Get environment variable from .env file
export $(grep -v '^#' .env | xargs)

# Set up the MLFlow server locally, bound to Supabase DB and S3
mlflow server \
  --backend-store-uri postgresql://postgres:$SUPABASE_POSTGRES_PASSWORD@$SUPABASE_POSTGRES_DB:$SUPABASE_POSTGRES_PORT/postgres \
  --host $MLFLOW_HOST \
  --port $MLFLOW_PORT