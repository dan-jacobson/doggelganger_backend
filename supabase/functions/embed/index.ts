import { env, pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2'
import { createClient } from 'jsr:@supabase/supabase-js@2'
import {
    ImageMagick,
    IMagickImage,
    initialize,
    MagickFormat,
    Magick,
  } from "https://deno.land/x/imagemagick_deno/mod.ts";

await initialize();

async function convertToPNG(inputBuffer: ArrayBuffer): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
        ImageMagick.read(inputBuffer, (image: IMagickImage) => {
            image.write(MagickFormat.Png, (data) => {
                resolve(data);
            });
        }, (error) => {
            reject(error);
        });
    });
}

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
        const formData = await req.formData();
        const imageFile = formData.get('image');

        if (!imageFile || !(imageFile instanceof File)) {
            return new Response('No valid image file found in the request', { status: 400 });
        }

        let embedding;
        try {
            const imageBuffer = await imageFile.arrayBuffer();
            const pngBuffer = await convertToPNG(imageBuffer);

            await ImageMagick.read(pngBuffer, async (img: IMagickImage) => {
                img.convert(MagickFormat.Rgba);
                const pixelData = await img.export();

                const rawImage = new RawImage(
                    new Uint8Array(pixelData),
                    img.width(),
                    img.height(),
                    img.channels()
                );

                const output = await pipe(rawImage);
                embedding = Array.from(output.data);
            });

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
