-- Two private buckets: "images" for raw uploads, "processedimages" for
-- model output. Every user is confined to their own folder
-- (<user_id>/<file>) in each bucket - no blanket USING(true)/WITH CHECK(true)
-- policy is ever added on top of these, since that combination (a scoped
-- policy plus a permissive one) is what caused the cross-tenant exposure in
-- the previous shared-project design.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('images', 'images', false, 52428800, array['image/png', 'image/jpeg']),
  ('processedimages', 'processedimages', false, 52428800, array['image/png', 'image/jpeg'])
on conflict (id) do nothing;

create policy "Users can read their own images"
on storage.objects
for select
to authenticated
using (bucket_id = 'images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can upload to their own images folder"
on storage.objects
for insert
to authenticated
with check (bucket_id = 'images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can update their own images"
on storage.objects
for update
to authenticated
using (bucket_id = 'images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can delete their own images"
on storage.objects
for delete
to authenticated
using (bucket_id = 'images' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can read their own processed images"
on storage.objects
for select
to authenticated
using (bucket_id = 'processedimages' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can upload to their own processed images folder"
on storage.objects
for insert
to authenticated
with check (bucket_id = 'processedimages' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can update their own processed images"
on storage.objects
for update
to authenticated
using (bucket_id = 'processedimages' and (storage.foldername(name))[1] = auth.uid()::text);

create policy "Users can delete their own processed images"
on storage.objects
for delete
to authenticated
using (bucket_id = 'processedimages' and (storage.foldername(name))[1] = auth.uid()::text);
