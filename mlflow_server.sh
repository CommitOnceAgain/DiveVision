# Set up the MLFlow server, bound to the Postgres DB and S3 configured via environment (Supabase in production, a local container in dev - see docker-compose.yml)
mlflow server \
  --backend-store-uri postgresql://$SUPABASE_POSTGRES_USER:$SUPABASE_POSTGRES_PASSWORD@$SUPABASE_POSTGRES_HOST:$SUPABASE_POSTGRES_PORT/$SUPABASE_POSTGRES_DB \
  --host $MLFLOW_HOST \
  --port $MLFLOW_PORT