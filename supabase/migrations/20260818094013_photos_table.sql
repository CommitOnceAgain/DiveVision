-- One row per uploaded photo, linking the raw and processed storage objects.
-- Kept intentionally minimal: this is meant to be cheap to maintain, not a
-- general-purpose asset system.

create table public.photos (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users (id) on delete cascade,
    original_path text not null,
    processed_path text,
    status text not null default 'processing' check (status in ('processing', 'completed', 'failed')),
    model_name text not null,
    created_at timestamptz not null default now()
);

create index photos_user_id_idx on public.photos (user_id);

alter table public.photos enable row level security;

create policy "Users can read their own photos"
on public.photos
for select
to authenticated
using (auth.uid() = user_id);

create policy "Users can insert their own photos"
on public.photos
for insert
to authenticated
with check (auth.uid() = user_id);

create policy "Users can update their own photos"
on public.photos
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can delete their own photos"
on public.photos
for delete
to authenticated
using (auth.uid() = user_id);
