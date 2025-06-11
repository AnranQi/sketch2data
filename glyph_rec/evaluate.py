from PIL import Image
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mode
import re





def dump_predicted_label(npy_dir, output_file): 
    uncertainty_dict = {}
    label_dict = {}
    # Loop through each category
    for c in range(num_of_category):
        print(f"Processing category {c}")
        predictions = []  # List to hold prediction arrays for the current category
        for i in range(num_training):
            file = f'{i}_para{c}.npy'
            npy_results = os.path.join(npy_dir, file)
            if not os.path.exists(npy_results):
                print(f"File {npy_results} does not exist.")
                continue
            npy_results = np.load(npy_results)
            predictions.append(npy_results)
        if not predictions:
            print(f"No predictions found for category {c}.")
            continue
        # Set the number of glyphs based on the first available file
        num_of_glyphs = predictions[0].shape[0]
        glyphs_dis = []  # List to hold average distances for each glyph
        most_common_labels = []  # List to hold most common labels for each glyph
        for i in range(num_of_glyphs):
            category_predictions = []  # Predictions for one glyph
            for j in range(num_training):
                pred = predictions[j][i]
                category_predictions.append(pred)
            pairwise_distance = pdist(category_predictions, metric='euclidean')
            average_distance = np.mean(pairwise_distance)
            glyphs_dis.append(average_distance)

            # Calculate the most common label for the current glyph
            pred_labels = [np.argsort(pred)[-1] for pred in category_predictions]
            most_common_label = mode(pred_labels).mode[0]
            most_common_labels.append(most_common_label)

        # Store uncertainty values and labels in the dictionaries
        uncertainty_dict[f'Uncertainty_{c}'] = glyphs_dis
        label_dict[f'Label_{c}'] = most_common_labels

    # Check if the number of glyphs is set
    if num_of_glyphs is None:
        print("No glyphs found. Exiting.")
    else:  
        # Create DataFrames from the dictionaries
        df_uncertainty = pd.DataFrame(uncertainty_dict)
        df_labels = pd.DataFrame(label_dict)
        # Concatenate the modified Series with the other DataFrames
        df_results =  pd.concat([img_paths.iloc[:num_of_glyphs, 0].reset_index(drop=True), df_uncertainty, df_labels], axis=1)
        df_results.columns = ['Glyph'] + df_results.columns.tolist()[1:]
        # Save the DataFrame to a CSV file
        df_results.to_csv(output_file, index=False)
        print(f'Uncertainty values and labels saved to file: {output_file}')


def reconstruction(csv_file_path, bounding_box_file, glyph_default_dir, save_reconstruction_name, design_size_x, design_size_y, glyph_size_x, glyph_size_y):
    # Load the CSV file
    csv_data = pd.read_csv(csv_file_path)

    # Strip the path to get only the image names from the Glyph column
    csv_data['Image'] = csv_data['Glyph'].apply(lambda x: x.split('\\')[-1])
    big_background = Image.new('RGBA', (design_size_x,design_size_y), 'white')



    # Read the content from the txt file
    with open(bounding_box_file, 'r') as file:
        lines = file.readlines()

    # Regular expression pattern to match the image name and bounding box coordinates
    pattern = r'Image: (\S+), Box: \(([\d.]+), ([\d.]+)\), \(([\d.]+), ([\d.]+)\)'

    # Iterate over each line in the file
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Extract image name and box coordinates
            image_name = match.group(1)
            x1, y1 = float(match.group(2)), float(match.group(3))
            x2, y2 = float(match.group(4)), float(match.group(5))
            match_row = csv_data[csv_data['Image'] == image_name]
            if not match_row.empty:
                labels = match_row[['Label_0', 'Label_1', 'Label_2']].values[0]
                labels = labels.tolist()
            else:
                print('something is wrong, check please')
            recognize_image  = os.path.join(glyph_default_dir, '_'.join(map(str, map(int, labels))) + '.png')
            recognize_image = Image.open(recognize_image).convert("RGBA")
            big_background.paste(recognize_image, (round(x1- ((glyph_size_x - (x2-x1)))/2), round(y1- ((glyph_size_y - (y2-y1)))/2)), recognize_image)  # top left

    big_background.save(save_reconstruction_name)
 





if __name__ == '__main__':
    #thoughts design##
    npy_dir = '.\\thoughts_rec_results'
    num_of_category = 3
    realdata_dir = "..\glyph_detection\\thoughts_detection_results"
    glyph_default_dir = ".\\thoughts_rec_dataset\default_glyph"
        
    csv_file = os.path.join(npy_dir, 'results.csv')
    annotations_file = os.path.join(realdata_dir, "test.csv")
    img_paths = pd.read_csv(annotations_file)
    num_training = 10 #number of training times

    save_reconstruction_name = '.\\thoughts_reconstruction.png'
    glyph_size_x = 130
    glyph_size_y = 130
    design_size_x = 1000#this is for part2
    design_size_y = 700

    bounding_box_file= os.path.join(realdata_dir,  'bounding_boxes.txt')

    dump_predicted_label(npy_dir, csv_file)
    reconstruction(csv_file, bounding_box_file, glyph_default_dir, save_reconstruction_name, design_size_x, design_size_y, glyph_size_x, glyph_size_y)
