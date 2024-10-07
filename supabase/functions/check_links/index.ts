import "jsr:@supabase/functions-js/edge-runtime.d.ts"
import { createClient } from 'jsr:@supabase/supabase-js@2'

const supabase = createClient(
  Deno.env.get("SUPABASE_URL") ?? "",
  Deno.env.get("SUPABASE_ANON_KEY") ?? ""
)

async function checkLink(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

async function checkImageUrl(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok && response.headers.get('Content-Type')?.startsWith('image/');
  } catch {
    return false;
  }
}

Deno.serve(async (req) => {
  try {
    // Generate a random vector
    const randomVector = Array.from({ length: 1536 }, () => Math.random());

    // Query 100 results from the database
    const { data: dogs, error } = await supabase.rpc('match_dogs', {
      query_embedding: randomVector,
      match_threshold: 0.0,
      match_count: 100
    });

    if (error) throw error;

    const results = await Promise.all(dogs.map(async (dog) => {
      const adoptionLinkValid = await checkLink(dog.adoption_link);
      const imageUrlValid = await checkImageUrl(dog.image_url);

      return {
        id: dog.id,
        name: dog.name,
        adoption_link: dog.adoption_link,
        adoption_link_valid: adoptionLinkValid,
        image_url: dog.image_url,
        image_url_valid: imageUrlValid
      };
    }));

    return new Response(
      JSON.stringify({ results }),
      { headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
})

/* To invoke locally:

  1. Run `supabase start` (see: https://supabase.com/docs/reference/cli/supabase-start)
  2. Make an HTTP request:

  curl -i --location --request POST 'http://127.0.0.1:54321/functions/v1/check_links' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0'

*/
