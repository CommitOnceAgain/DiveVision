-- Written exclusively by the FastAPI backend (via the service-role key, which
-- never leaves the backend) from the /leaderboard endpoint. No client-facing
-- policies are defined, so RLS denies every request from the anon/authenticated
-- roles by default; a future read endpoint can add a scoped SELECT policy
-- when one is actually needed.

create table public.leaderboard (
    id uuid primary key default gen_random_uuid(),
    model_name text not null,
    dataset_name text not null,
    metric_name text not null,
    score double precision not null,
    created_at timestamptz not null default now()
);

alter table public.leaderboard enable row level security;
