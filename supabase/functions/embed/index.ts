import { env, pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
import { createClient } from 'jsr:@supabase/supabase-js@2'
import { multiParser, Form, FormFile } from 'https://deno.land/x/multiparser@v2.1.0/mod.ts'
import { decode } from "https://deno.land/x/imagescript@1.3.0/mod.ts";

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
Deno.serve(async (request) => {
    if (request.method !== 'POST') {
        return new Response('Method Not Allowed', { status: 405 });
    }

    try {
        const form = await request.formData();
        const imageFile = await form.get("image");

        if (!imageFile || !(imageFile instanceof File)) {
            return new Response('No valid image file found in the request', { status: 400 });
        }

        let embedding;
        try {
            const arrayBuffer = await imageFile.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);
            const image = await decode(uint8Array);

            const rawImage = new RawImage(
                new Uint8Array(image.bitmap),
                image.width,
                image.height,
                image.channels as 1|2|3|4
            );

            const output = await pipe(rawImage);
            embedding = Array.from(output.data);
            
            return new Response(
                JSON.stringify({ 
                    message: 'Image processed successfully',
                    embedding: embedding
                }),
                { headers: { 'Content-Type': 'application/json' } }
            );
        } catch (imageError) {
            console.error('Error processing image:', imageError);
            return new Response('Error processing image: ' + imageError.message, { status: 400 });
        }
    } catch (error) {
        console.error('Error processing request:', error);
        return new Response('Internal Server Error: ' + error.message, { status: 500 });
    }
});
