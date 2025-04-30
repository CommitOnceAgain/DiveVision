drop index if exists "storage"."idx_name_bucket_level_unique";

CREATE UNIQUE INDEX idx_name_bucket_unique ON storage.objects USING btree (name COLLATE "C", bucket_id);

grant delete on table "storage"."prefixes" to "postgres";

grant insert on table "storage"."prefixes" to "postgres";

grant references on table "storage"."prefixes" to "postgres";

grant select on table "storage"."prefixes" to "postgres";

grant trigger on table "storage"."prefixes" to "postgres";

grant truncate on table "storage"."prefixes" to "postgres";

grant update on table "storage"."prefixes" to "postgres";

grant delete on table "storage"."s3_multipart_uploads" to "postgres";

grant insert on table "storage"."s3_multipart_uploads" to "postgres";

grant references on table "storage"."s3_multipart_uploads" to "postgres";

grant select on table "storage"."s3_multipart_uploads" to "postgres";

grant trigger on table "storage"."s3_multipart_uploads" to "postgres";

grant truncate on table "storage"."s3_multipart_uploads" to "postgres";

grant update on table "storage"."s3_multipart_uploads" to "postgres";

grant delete on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant insert on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant references on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant select on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant trigger on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant truncate on table "storage"."s3_multipart_uploads_parts" to "postgres";

grant update on table "storage"."s3_multipart_uploads_parts" to "postgres";

create policy "Enable insert for authenticated users only"
on "storage"."objects"
as permissive
for insert
to authenticated
with check (true);


create policy "Enable select for authenticated users only"
on "storage"."objects"
as permissive
for select
to authenticated
using (true);


create policy "Give users access to own folder 1ffg0oo_0"
on "storage"."objects"
as permissive
for select
to authenticated
using (((bucket_id = 'images'::text) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[1])));


create policy "Give users access to own folder 1ffg0oo_1"
on "storage"."objects"
as permissive
for insert
to authenticated
with check (((bucket_id = 'images'::text) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[1])));


create policy "Give users access to own folder 1ffg0oo_2"
on "storage"."objects"
as permissive
for update
to authenticated
using (((bucket_id = 'images'::text) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[1])));


create policy "Give users access to own folder 1ffg0oo_3"
on "storage"."objects"
as permissive
for delete
to authenticated
using (((bucket_id = 'images'::text) AND (( SELECT (auth.uid())::text AS uid) = (storage.foldername(name))[1])));



