import os

import argparse

from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np

from pillow_heif import register_heif_opener
register_heif_opener()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model_name = "microsoft/Florence-2-large"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, trust_remote_code=True).to(
    device).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def run_model(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                          max_new_tokens=512, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer


def get_bboxes_person_head(image):
    person = run_model('<CAPTION_TO_PHRASE_GROUNDING>', image, "person")
    if len(person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']) > 1:  # chooce max area
        areas = [(bb[2]-bb[0]) * (bb[3]-bb[1]) for bb in person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']]
        person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'] = [person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'][np.argmax(areas)]]
    person_bbox = person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    head = run_model('<CAPTION_TO_PHRASE_GROUNDING>', image, "head")
    if len(person['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']) == 0:
        return [], [head[0]]
    if len(head['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']) > 1:  # chooce smallest inside person box
        centers = [[(bb[2]+bb[0])/2, (bb[3]+bb[1])/2] for bb in head['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']]
        new_bboxes = []
        for i in range(len(centers)):
            if person_bbox[0][0] <= centers[i][0] <= person_bbox[0][2] and person_bbox[0][1] <= centers[i][1] <= person_bbox[0][3]:
               new_bboxes.append(head['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'][i])
        if len(new_bboxes) > 1:
            areas = [(bb[2]-bb[0]) * (bb[3]-bb[1]) for bb in new_bboxes]
            new_bboxes = [new_bboxes[np.argmin(areas)]]
        head['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'] = new_bboxes
    head_bbox = head['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    if len(head_bbox) == 0:
        return person_bbox, []
    return person_bbox, head_bbox


def rotate_if_needed(image, person_bbox, head_bbox):
    if isinstance(person_bbox, dict): person_bbox = person_bbox['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    if isinstance(head_bbox, dict): head_bbox = head_bbox['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    # x1, y1, x2, y2 = bbox
    width, height = image.width, image.height
    closest_side = [abs(ph[0] - ph[1]) for ph in zip(person_bbox[0], head_bbox[0])]

    smolest_idx = 1
    smolest_side = 1000
    for i, cs in enumerate(closest_side):
        if cs < smolest_side:
            smolest_side = cs
            smolest_idx = i

    if closest_side[1] / min(width, height) < 0.03:
        smolest_idx = 1
    angle = [-1, 0, 1, 2][smolest_idx]

    if angle == -1:
        person_bbox = [[height - person_bbox[0][3], person_bbox[0][0], height - person_bbox[0][1], person_bbox[0][2]]]
        head_bbox = [[height - head_bbox[0][3], head_bbox[0][0], height - head_bbox[0][1], head_bbox[0][2]]]
    elif angle == 1:
        person_bbox = [[person_bbox[0][1], width - person_bbox[0][2], person_bbox[0][3], width - person_bbox[0][0]]]
        head_bbox = [[head_bbox[0][1], width - head_bbox[0][2], head_bbox[0][3], width - head_bbox[0][0]]]
    elif angle == 2:
        person_bbox = [[width - person_bbox[0][2], height - person_bbox[0][3], width - person_bbox[0][0],
                        height - person_bbox[0][1]]]
        head_bbox = [
            [width - head_bbox[0][2], height - head_bbox[0][3], width - head_bbox[0][0], height - head_bbox[0][1]]]

    if angle != 0:
        return Image.fromarray(np.rot90(image, k=angle)), person_bbox, head_bbox
    return image, person_bbox, head_bbox


def crop_resize(image, size, person_bbox=None, head_bbox=None):
    width, height = image.width, image.height
    aspect = height / width
    margin = int(min(height, width) * 0.05)
    bbox = person_bbox if person_bbox is not None else head_bbox

    if aspect > 1:  # tall
        cut_size = (aspect - 1) * width
        if height - bbox[0][3] > cut_size + margin:
            # left, top, right, bottom
            image = image.crop([0, 0, width, height - int(cut_size)])
        else:
            bot = int(max(0, height - bbox[0][3] - margin))
            top = cut_size - bot
            if top > bbox[0][1] - margin:
                top = bbox[0][1] - margin
                bot = cut_size - top

            image = image.crop([0, top, width, height - bot])

    elif aspect < 1:  # wide
        cut_size = (1 - aspect) * width
        left_space = max(0.001, bbox[0][0] - margin)
        right_space = max(0.001, width - bbox[0][2] - margin)
        left_cut = int(left_space / (left_space + right_space) * cut_size)
        right_cut = width - height - left_cut
        image = image.crop([left_cut, 0, width - right_cut, height])

    return image.resize((size, size), Image.Resampling.LANCZOS)



def process_directory(args, image_dir, total_to_process: int = None, ):
    total_processed = 0
    if len(args.output_dir) > 0:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = image_dir

    contents = [fl for fl in os.listdir(image_dir) if
                (os.path.splitext(fl)[-1].lower() in [".jpg", ".png", ".jpeg", ".heic"] or os.path.isdir(fl))]
    for filename in tqdm(contents, desc="Processing Images"):
        full_filepath = os.path.join(image_dir, filename)
        if os.path.isdir(full_filepath):
            process_directory(args, full_filepath)

        elif filename.lower().endswith((".jpg", ".png", ".jpeg", ".heic")):
            try:
                # convert heic to jpg
                if filename.lower().endswith(".heic"):
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_filepath = os.path.join(output_dir, new_filename)
                    image = Image.open(full_filepath)
                    image = image.convert("RGB")
                    image.save(new_filepath, "JPEG", quality=95)
                    os.remove(full_filepath)
                    filename = new_filename
                else:
                    image = Image.open(full_filepath)

                person, head = get_bboxes_person_head(image)
                image_rotated, person_bbox, head_bbox = rotate_if_needed(image, person, head)
                image_resized = crop_resize(image_rotated, args.target_size, person_bbox, head_bbox)
                caption = run_model("<MORE_DETAILED_CAPTION>", image_resized)['<MORE_DETAILED_CAPTION>']

                if len(args.trigger) > 0:
                    caption = args.trigger + " " + caption

                output_filepath = os.path.join(output_dir, filename)
                image_resized.save(output_filepath, format='JPEG', quality=95)
                # save in textfile
                with open(os.path.splitext(output_filepath)[0] + ".txt", 'w') as f:
                    f.write(caption)

                total_processed += 1
                if total_to_process is not None and total_processed >= total_to_process:
                    break

            except Exception as e:
                import traceback

                if "CUDA error" in str(e):
                    import sys
                    sys.exit(1)



def parse_args():
    parser = argparse.ArgumentParser(description="Process images and generate captions.")
    parser.add_argument("input_dir", type=str, help="Directory containing the images.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save images and captions.")
    parser.add_argument("--trigger",type=str,default="[trigger]",help="Trigger word/s for character",)
    parser.add_argument("--target_size",type=int,default=1024,help="Side size for image (squared)",)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    print("Trigger word will be added to captions:", args.trigger)
    process_directory(args, args.input_dir)
