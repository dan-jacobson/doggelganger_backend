import { env, pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
import { createClient } from 'jsr:@supabase/supabase-js@2'
import {
    ImageMagick,
    IMagickImage,
    initialize,
    MagickFormat,
  } from "https://deno.land/x/imagemagick_deno/mod.ts";

await initialize();

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
        // Parse the multipart form data
        const formData = await req.formData();
        const imageFile = formData.get('image');

        if (!imageFile || !(imageFile instanceof File)) {
            return new Response('No valid image file found in the request', { status: 400 });
        }
        
        const arrayBuffer = await imageFile.arrayBuffer();
        const data = new Uint8Array(arrayBuffer);

        await ImageMagick.read(data, async (img: IMagickImage) => {
            const width = img.width();
            const height = img.height();
            const channels = img.channels();

            // Convert image to RGBA format
            img.convert(MagickFormat.Rgba);
            const pixelData = await img.export();

            const rawImage = new RawImage(
                new Uint8Array(pixelData),
                width,
                height,
                channels
            );

            // Generate the embedding
            const output = await pipe(rawImage);

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
