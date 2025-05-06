// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { Database } from "../../database.types.ts";

console.log("Hello from 'image-transform' function!");

// API Endpoint for Image Processing
const ApiEndpointUrl = Deno.env.get("API_ENDPOINT");
if (ApiEndpointUrl === undefined) {
  throw new Error("API_ENDPOINT environment variable not found!");
}
// Define the expected type of record
type SoRecord = Database["storage"]["Tables"]["objects"]["Row"];
// Webhook definition
interface WebhookPayload {
  type: "INSERT";
  table: string;
  record: SoRecord;
  schema: "public";
  old_record: null | SoRecord;
}

Deno.serve(async (req) => {
  const payload: WebhookPayload = await req.json();
  const soRecord = payload.record;

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
    ).select("bucket_id, name, path_tokens").eq("id", soRecord.id);
  if (status) throw status;

  console.log(`Storage info retrieved: ${JSON.stringify(storageInfo)}`);
  // Destructuring storage information
  const bucket = storageInfo[0].bucket_id!;
  const path = storageInfo[0].path_tokens?.join("/")!;
  const name = storageInfo[0].name!;

  // Construct image url from storage
  const { data: url, error } = await supabaseAdminClient.storage.from(
    bucket,
  )
    .createSignedUrl(
      path,
      60,
    );
  if (error) throw error;
  // Download image as blob
  const image = await fetch(url.signedUrl);
  const imageBlob = await image.blob();

  console.log(`Fetched image (size=${imageBlob.size})`);

  // Set up data to be sent via HTTP request
  const imageBody = new FormData();
  imageBody.append("file", imageBlob, name);

  console.log(imageBody);

  // Make POST request to the API endpoint for image processing
  const request = new Request(ApiEndpointUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: imageBody,
  });
  const processedImage = await fetch(request);
  console.log(request.url);
  console.log();

  // Save received (!) image to Supabase storage
  supabaseAdminClient.storage.from(
    "processedimages",
  ).upload(path, processedImage.body!).catch((error) => console.log(error));
  console.log("Uploaded processed image");

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
