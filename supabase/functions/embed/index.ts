import { env, pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
import { createClient } from 'jsr:@supabase/supabase-js@2'
import sharp from 'npm:sharp@0.32.6'

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

const loadImageFunction = async (/**@type {sharp.Sharp}*/img) => {
    const metadata = await img.metadata();
    const rawChannels = metadata.channels;

    let { data, info } = await img.rotate().raw().toBuffer({ resolveWithObject: true });

    const newImage = new RawImage(new Uint8ClampedArray(data), info.width, info.height, info.channels);
    if (rawChannels !== undefined && rawChannels !== info.channels) {
        // Make sure the new image has the same number of channels as the input image.
        // This is necessary for grayscale images.
        newImage.convert(rawChannels);
    }
    return newImage;
}

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

        // Convert the File to a Buffer
        const imageBuffer = await imageFile.arrayBuffer();

        // Use sharp to process the image
        const sharpImage = sharp(imageBuffer);

        // Create a RawImage using the new loadImageFunction
        const img = await loadImageFunction(sharpImage);
        
        // Generate the embedding
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
