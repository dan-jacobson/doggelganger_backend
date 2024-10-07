import "jsr:@supabase/functions-js/edge-runtime.d.ts"
import { createClient } from 'jsr:@supabase/supabase-js@2'

const supabase = createClient(
  Deno.env.get("SUPABASE_URL") ?? "",
  Deno.env.get("SUPABASE_ANON_KEY") ?? ""
)

async function checkLink(url: string, isRetry = false, maxRetries = 3): Promise<boolean> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, { method: 'HEAD', redirect: 'follow' });
      return response.ok;
    } catch {
      if (attempt === maxRetries - 1 || isRetry) return false;
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000 + Math.random() * 1000));
    }
  }
  return false;
}

async function checkImage(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD', redirect: 'follow' });
    return response.ok && response.headers.get('Content-Type')?.startsWith('image/');
  } catch {
    return false;
  }
}

Deno.serve(async (req) => {
  try {
    const randomHash = Array.from({length: 16}, () => Math.floor(Math.random() * 16).toString(16)).join('');

    let { data: dogs, error } = await supabase
      .from('dog_embeddings')
      .select()
      .gt('id', randomHash)
      .order('id')
      .limit(100);

    if (error) throw error;

    if (dogs.length < 100) {
      const remainingCount = 100 - dogs.length;
      const { data: moreDogs, error: moreError } = await supabase
        .from('dog_embeddings')
        .select()
        .lte('id', randomHash)
        .order('id')
        .limit(remainingCount);

      if (moreError) throw moreError;
      dogs = dogs.concat(moreDogs);
    }

    const failedDogs = [];

    for (const dog of dogs) {
      let adoptionValid = await checkLink(dog.adoption_link);
      if (!adoptionValid) {
        adoptionValid = await checkLink(dog.adoption_link, true);
      }

      const imageValid = await checkImage(dog.image_url);

      if (!adoptionValid || !imageValid) {
        failedDogs.push(dog.id);
      }
    }

    const failureRate = failedDogs.length / dogs.length;

    if (failureRate <= 0.33) {
      // Update the database by removing failed dogs
      const { error: deleteError } = await supabase
        .from('dog_embeddings')
        .delete()
        .in('id', failedDogs);

      if (deleteError) throw deleteError;

      return new Response(
        JSON.stringify({ 
          totalChecked: dogs.length, 
          failedCount: failedDogs.length, 
          failureRate: failureRate,
          dogsRemoved: failedDogs
        }),
        { headers: { "Content-Type": "application/json" } }
      );
    } else {
      // More than 33% failed, don't remove any dogs
      return new Response(
        JSON.stringify({ 
          totalChecked: dogs.length, 
          failedCount: failedDogs.length, 
          failureRate: failureRate,
          message: "More than 33% of dogs failed checks. No dogs were removed from the database."
        }),
        { headers: { "Content-Type": "application/json" } }
      );
    }
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
