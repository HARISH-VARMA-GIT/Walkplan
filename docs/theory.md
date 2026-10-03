1) Requirement :prompt architectures, tool use, routing, evals
2) Xactimate uses floor plan and sketches to generate a final insurance claim report  -- 1hr video
    Various methods to generate a floor plan:
    Method 1: Uploading and Scaling an Image Underlay (PNG, JPEG)
    Method 2: Importing a Structured Template (Matterport TruePlan .SKX)
    Method 3: Importing via Integrations (Encircle or Magicplan) uses .ESX

3) Magic plan already does the method 3 where it has app which can take images
    Competitors: **Magic Plan, Encircle Floor Plan**
4) The schema of .ESX and .SKX is private/ proprietary it belongs to Verisk company
    Two ways to get the format right

    **Official**: Join Verisk's partner program to get the spec. You apply, sign an NDA, meet security and insurance requirements and give client references. It's slower, but it's the clean route for a commercial product.

    **Reverse-engineer** it from sample files: Faster, but Xactimate's license bars reverse engineering or decoding its software, and copying its price data (§3.5, §13.9, §14.1). Verisk can also change the format and break your imports. I'm not a lawyer, so get legal advice before you ship this way.

        The legality of reverse engineering a file format for the sole purpose of interoperability—enabling your independent software program to output files compatible with another platform—is a well-established legal principle, particularly under United States law

        You are not copying Xactimate's proprietary source code, bypassing their DRM (Digital Rights Management) to steal the software, or cloning their app. 


-------------------------------------

1) MoGe: Accurate Monocular Geometry Estimation --- Creates the mesh,point cloud,depth

2) Grounding dino sam --  Detects the objecs and masks
    One can combine Grounding DINO with the Segment Anything model for text-based mask generation as introduced in Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks. 