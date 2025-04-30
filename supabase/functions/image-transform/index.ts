// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { Database } from "../../database.types.ts";

console.log("Hello from 'image-transform' function!");

const API_SERVER_URL = "http://localhost:8000";

type SoRecord = Database["storage"]["Tables"]["objects"]["Row"];
interface WebhookPayload {
  type: "INSERT" | "UPDATE" | "DELETE";
  table: string;
  record: SoRecord;
  schema: "public";
  old_record: null | SoRecord;
}

Deno.serve(async (req) => {
  // const payload: WebhookPayload = await req.json();
  // const soRecord = payload.record;

  const id = "4b1205b8-183a-4a17-9ba8-d371e5b7c44e";
  const bucket_id = "images";

  const supabaseAdminClient = createClient<Database>(
    // Supabase API URL - env var exported by default when deployed.
    Deno.env.get("SUPABASE_URL") ?? "",
    // Supabase API SERVICE ROLE KEY - env var exported by default when deployed.
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
  );

  console.log("Supabase Admin Client OK");

  const { data: storageInfo, error: status } = await supabaseAdminClient
    .schema(
      "storage",
    )
    .from(
      "objects",
    ).select("path_tokens").eq("id", id);
  if (status) throw status;

  console.log(`Storage info retrieved: ${storageInfo}`);

  // Construct image url from storage
  const { data: url, error } = await supabaseAdminClient
    .storage
    .from(bucket_id!)
    .createSignedUrl(storageInfo!.join("/"), 60);
  if (error) throw error;
  const signedUrl = url.signedUrl;
  const imageData = new FormData();
  // Download the image, and append it as HTTP body
  imageData.append("image", await (await fetch(signedUrl)).blob());

  // // Run image transformation
  // await fetch(`${API_SERVER_URL}/image/`, { method: "POST", body: imageData })
  //   .then((response) => response.blob())
  //   .then((data) => {
  //     // Store image caption in Database table
  //     supabaseAdminClient
  //       .from("images")
  //       .insert({ id: soRecord.id!, processedimageid: data })
  //       .throwOnError();
  //   })
  //   .catch((error) => console.error(error));

  return new Response("ok");
});

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/image-transform' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0' \
    --header 'Content-Type: application/json' \
    --data '{"name":"Functions"}'

*/
