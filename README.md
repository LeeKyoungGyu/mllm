# Optical Illusion in VLM
---
### 📋 Overview
This project investigates the fascinating intersection of computer vision and cognitive psychology by testing how advanced MLLMs (Multimodal Large Language Models) interpret optical illusions. We compare multiple cutting-edge models to understand their visual perception capabilities and limitations.

### ⚙️ VLMs
| VLMs | Version | Parameters |
|--------|------------------|---------|
|LLava|v1.5|7b|
|LLava|v1.5|13b|
|LLava|v1.6|7b|
|InternVL | | |
|Qwen| | |
|BLIP| | |
| | | |
| | | |

### 🖼️ Optical Illusions Dataset Classification
![Image](https://raw.githubusercontent.com/LeeKyoungGyu/mllm/main/images.png)


| File Name | Illusion Name | Illusion Type | Main Effect | Description |
|-----------|---------------|---------------|-------------|-------------|
| adelson1-3 | Adelson's Checker Shadow | Brightness Illusion | Same grey looks different | Illusion caused by shadows and checkerboard pattern altering perceived brightness |
| cafe1-4 | Café Wall Illusion | Geometric Illusion | Parallel lines appear tilted | Line distortion due to offset black-and-white tile patterns |
| cube1-3 | Necker Cube | Ambiguous Figure | 3D orientation shifts | Wireframe cube perceived in multiple depth orientations |
| duckrabbit1-3 | Duck-Rabbit Illusion | Ambiguous Figure | Two different animals perceived | Image alternates between duck and rabbit |
| ebbinghaus2-4 | Ebbinghaus Illusion | Size Illusion | Central circle size distortion | Perceived size of the central circle affected by surrounding circles |
| ehrenstein1-3 | Ehrenstein Illusion | Subjective Contours | Non-existent shapes perceived | Illusionary square created by radial lines |
| hering1-3 | Hering Illusion | Geometric Illusion | Straight lines appear curved | Parallel lines distorted by radial background patterns |
| impossible1-4 | Impossible Objects | Paradox Illusion | Impossible 3D structures | Penrose triangle, impossible staircase |
| kanizsa1-5 | Kanizsa Triangle | Subjective Contours | Illusionary triangle perceived | White triangle formed by "Pac-Man" shapes |
| muller1-5 | Müller-Lyer Illusion | Length Illusion | Line length distortion by arrows | Arrowheads influencing perceived line length |
| neon1-3 | Neon Color Spreading | Color Illusion | Color appears to spread | Colors perceived to spread beyond boundaries |
| penrose1-3 | Penrose Stairs/Triangle | Paradox Illusion | Infinite loop structure | Escher's impossible stairs and triangles |
| poggendorff1-3 | Poggendorff Illusion | Geometric Illusion | Distorted line segment continuity | Line segment continuity hidden by parallel lines |
| ponzo1-3 | Ponzo Illusion | Size Illusion | Size distortion due to perspective | Same-sized bars perceived differently due to converging lines |
| rubbins1-3 | Rubin's Vase | Ambiguous Figure | Figure-ground reversal | Vase and faces alternately perceived |
| strawberris1-3 | Impossible Colors | Color Illusion | Impossible color combinations | Perceived impossible colors due to color contrast and afterimages |
| verticle1-4 | Vertical-Horizontal Illusion | Length Illusion | Vertical lines perceived longer | Vertical and horizontal lines of equal length perceived differently |
| wundt1-3 | Wundt Illusion | Geometric Illusion | Curves perceived as straight lines | Symmetrical curved patterns creating illusions |
| zollner1-3 | Zöllner Illusion | Geometric Illusion | Parallel lines appear tilted | Parallel lines distorted by diagonal hatching |



### 📊 Results Preview
---

