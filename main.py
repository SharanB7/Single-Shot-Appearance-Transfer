from PIL import Image
from utils import Utils
import os

def process_inputs(final_image_desc: str, folder_path: str):

    utils_obj = Utils()

    for image_filename in os.listdir(folder_path):
        utils_obj.input_name = image_filename
        image_path = os.path.join(folder_path, image_filename)
        print("Image Path:")
        print(image_path)
        utils_obj.llm_call(final_image_desc, image_path)
    
        new_row = {'input_image': utils_obj.input_name, 'output_image': utils_obj.output_name, 'similarity': utils_obj.image_similarity_score(utils_obj.input_embedding,utils_obj.output_image)}
        utils_obj.df.loc[len(utils_obj.df)] = new_row

    print("Similarity Scores Generated")
    utils_obj.df.to_csv('similarity_results.csv',index=False)

if __name__ == "__main__":
    final_image_desc = "Final target image description : 'A dog wearing a hat that has a style of dress'"
    folder_path = "Dataset"
    
    process_inputs(final_image_desc, folder_path)