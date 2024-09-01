import { env, pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
import { createClient } from 'jsr:@supabase/supabase-js@2'

// Because the tutorial said to
env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;

// const supabase = createClient(
//   Deno.env.get("SUPABASE_URL"),
//   Deno.env.get("SUPABASE_ANON_KEY")
// )

// Construct pipeline outside of serve for faster warm start
const pipe = await pipeline(
    'image-feature-extraction',
    'xenova/dinov2-small'
)

// Deno Handler
Deno.serve(async (req) => {
    if (req.method !== 'POST') {
        return new Response('Method Not Allowed', { status: 405 });
    }

    try {
        // Parse the request body
        im = await req.blob();
        img = RawImage.fromBlob(im)
        
        // Generate the selfie embedding directly from the ArrayBuffer
        const output = await pipe(img);

        // Get embedding
        const embedding = Array.from(output.data);

        // Find most similar dog in our db
        // const { dog, error } = await supabase.rpc(
        // "find_most_similar_dog",
        // {
        //   embedding,
        // }
        // );

        // Return the record
        return new Response(
            JSON.stringify({ 
                message: 'Image processed successfully',
                embedding: embedding
            }),
            { headers: { 'Content-Type': 'application/json' } }
        );
    } catch (error) {
        console.error('Error processing request:', error);
        return new Response('Internal Server Error', { status: 500 });
    }
});
