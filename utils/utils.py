import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline, AutoPipelineForText2Image

from ip_adapter import IPAdapterXL

import json
from peft import PeftModel
from PIL import Image
import torch.nn.functional as F

import requests
import os
import re
import pandas as pd



class Utils:
    def __init__(self, device="cuda:0"):
        
        # Kandinsky model
        self.image_count = 1
        self.prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16)
        self.pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16)

        self.clip_vision_model = self.prior_pipeline.image_encoder
        self.preprocess = self.prior_pipeline.image_processor

        
        # IP Adapter
        self.sd_pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        self.sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        self.sd_pipeline.set_ip_adapter_scale(0.6)

        self.device = device

        self.results = { 'V0': '', 'T0': '', 'T1': '', 'I0': '', 'I2': '' }

        self.llm_response = ''
        self.input_embedding = None
        self.output_image = None
        self.output_name = None
        self.input_name = None
        self.df = pd.DataFrame(columns=['input_image', 'output_image', 'similarity'])
        
    def clip_encode_text(self, text):
        negative_prompt = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
        self.prior_pipeline.to(self.device)
        image_embeds2, _ = self.prior_pipeline(text, negative_prompt, guidance_scale=1.0, num_inference_steps=100, generator=torch.Generator().manual_seed(42)).to_tuple()
        return image_embeds2

    def clip_encode_image(self, image):

        inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
        print("Processed image tensor shape:", inputs['pixel_values'].shape)
        self.clip_vision_model.to("cuda:0")
        with torch.no_grad():
            outputs = self.clip_vision_model(pixel_values=inputs['pixel_values'])
            image_features = outputs.image_embeds
        return image_features

    def text2image(self, image_embeddings):
        _, negative_image_embeds =  self.prior_pipeline("", "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality", guidance_scale=2.0, num_inference_steps=20, generator=torch.Generator().manual_seed(42)).to_tuple()
        self.pipeline.to(self.device)
        image = self.pipeline(image_embeds=image_embeddings, negative_image_embeds=negative_image_embeds, height=768, width=768, generator=torch.Generator().manual_seed(42)).images[0]
        return image 
    
    def image_similarity_score(self, embedding1, image2):
        embedding2 = self.clip_encode_image(image2)

        # Ensure both embeddings are on the same device and of the same shape
        embedding1 = embedding1.to(self.device)
        embedding2 = embedding2.to(self.device)

        # Step 2: Normalize the embeddings along the embedding dimension
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)

        # Step 3: Compute cosine similarity
        # Since these are normalized, their dot product is the cosine similarity
        similarity = torch.mm(embedding1, embedding2.T).item()

        return similarity


    def ipadapter_text2image(self, text, image=None):
        self.sd_pipeline.to(self.device)
        images = self.sd_pipeline(
            prompt=text,
            ip_adapter_image=image,
            negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            generator=torch.Generator().manual_seed(42),    
        ).images

        image = images[0]
        if isinstance(image, Image.Image):
            folder = "final"
            os.makedirs(folder, exist_ok=True)
            image_filename = f"final_image{self.image_count}.jpg"
            image_path = os.path.join(folder, image_filename)
            image.save(image_path)
            self.output_image = image
            self.output_name = image_filename

            embedding_folder = "embeddings"
            embeddings_filename = f"final_embeddings{self.image_count}.json"
            embeddings_path = os.path.join(embedding_folder, embeddings_filename)
            for key, value in self.results.items():
                print(key, type(value))

            data_serialized = {k: v.tolist() for k, v in self.results.items() if k != 'Generated_Image'}
            with open(embeddings_path, "w") as file:
                json.dump(data_serialized, file, indent=4)

            self.image_count += 1
        return images[0]

    def encode_image(self, image_features):
        return self.clip_encode_image(image_features)

    def encode_text(self, phrase):
        return self.clip_encode_text(phrase)

    def subtract_embeddings(self, embedding1, embedding2):
        return embedding1 - embedding2

    def add_embeddings(self, embedding1, embedding2):
        return embedding1 + embedding2

    def generate_image_from_embedding(self, embedding):
        self.input_embedding = embedding
        image = self.text2image(embedding)
        if isinstance(image, Image.Image):
            output_folder = "inter"
            os.makedirs(output_folder, exist_ok=True)
            image_filename = f"inter_image{self.image_count}.jpg"
            image_path = os.path.join(output_folder, image_filename)
            image.save(image_path)
        return image


    def refine_image_with_phrase(self, image, phrase):
        return self.ipadapter_text2image(phrase, image)

    def process_json(self, tasks, image_path):
        
        # image encoding
        image = Image.open(image_path)
        self.results['V0'] = self.clip_encode_image(image)
        print("image encodings", self.results['V0'])

        print("tasks", tasks)


        for task in tasks:
            task_type = task.get("function")
            action = task.get("action")
            input_text = task.get("input_text")
            input_variables = task.get("input_variables")
            output_variable = task.get("output_variable")

            print(task_type, action, input_text, input_variables, output_variable)
            

            if task_type == "clip_encode_text":
                output = self.encode_text(input_text[0])
                print("clip_encode_text", output)
                self.results[output_variable] = output


            elif task_type == "compute":
                var1, var2 = input_variables[0], input_variables[1]
                if action == "subtract":
                    output = self.subtract_embeddings(self.results[var1], self.results[var2])
                    print("subtraction", output)
                    self.results[output_variable] = output

                elif action == "add":
                    output = self.add_embeddings(self.results[var1], self.results[var2])
                    self.results[output_variable] = output
                    print("addition", output)
                    print(self.results)

            elif task_type == "text2image":
                input_var = input_variables[0]
                output = self.generate_image_from_embedding(self.results[input_var])
                print("text_2_image", output)
                self.results[output_variable] = output

            elif task_type == "ipadapter_text2image":
                input_image = self.results['Generated_Image']
                input_phrase = input_text[0]
                output = self.refine_image_with_phrase(input_image, input_phrase)
        

    def remove_extra_spaces(self, generated_answer):
        cleaned = re.sub(r'(?<!\w) +| +(?!\w)', '', generated_answer)
        cleaned = re.sub(r'[\t\n\r\f\v]+', '',cleaned)
        return cleaned

    def extract_json(self, json_text):
        json_pattern = r"\[\{.*"
        incomplete_json = re.findall(json_pattern, json_text)
        if not incomplete_json:
            raise ValueError("No JSON found matching the pattern.")
        return incomplete_json[-1]

    def balance_json(self, incomplete_json):
        stack = []
        balanced_json = incomplete_json

        for char in incomplete_json:
            if char in "[{":
                stack.append(char)
            elif char in "]}":
                if stack and (
                    (char == "]" and stack[-1] == "[")
                    or (char == "}" and stack[-1] == "{")
                ):
                    stack.pop()
                else:
                    stack.append(char)

        for unmatched in reversed(stack):
            if unmatched == "[":
                balanced_json += "]"
            elif unmatched == "{":
                balanced_json += "}"

        return balanced_json

    def parse_json(self, balanced_json):
        return json.loads(balanced_json)[0]

    def clean_output(self, generated_answer):
        json_text = self.remove_extra_spaces(generated_answer)
        incomplete_json = self.extract_json(json_text)
        balanced_json = self.balance_json(incomplete_json)
        final_json = self.parse_json(balanced_json)
        return final_json

    def format_question(self, question, system_message):
        return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"
    
    def llm_call(self, final_image_desc, image_path):
        # URL of the endpoint
        url = "http://3.142.196.92/get-response"
        auth_token = "613534398485e170652d4bda1aca3931"
        # Headers for the request
        headers = {
            "Authorization": auth_token
        }


        # Parameters to send with the GET request
        params = {
        "prompt": final_image_desc
        }

        try:
        # Make the GET request to the endpoint
            if self.llm_response != '':
                self.process_json(self.llm_response, image_path)
            else:
                response = requests.get(url, params=params, headers=headers)

                # Check if the request was successful
                if response.status_code == 200:
                    # Parse and use the JSON response
                    data = response.json()
                    self.llm_response = data
                    print("Response from the endpoint:")
                    print(data)
                    self.process_json(data, image_path)
                else:
                    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
                    print(f"Error Message: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    utils = Utils("cuda:0")
    prompt = "A dog wearing a hat in the style of the dress."
    image_path = "/home/cr8dl-user/abhiram/Zero-shot-Appearance-Transfer-using-CLIP-Abstraction/red_dress.jpg" 
    
    from PIL import Image
    img = Image.open(image_path)
    utils.text2img.to("cuda:0")

    text_embeds, last_hidden_state = utils.clip_encode_text(prompt)
    img = utils.text2image(last_hidden_state)
    img.save("output.jpg")