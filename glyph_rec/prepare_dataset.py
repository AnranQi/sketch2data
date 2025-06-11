import os
import pandas as pd
import random
import glob


def split_data(image_dir, train_split=0.7, val_split=0.1, test_split=0.2):
    import random,shutil
    images = sorted(os.listdir(image_dir))
    random.shuffle(images)

    total_count = len(images)
    train_count = int(total_count * train_split)
    val_count = int(total_count * val_split)

    train_data = images[:train_count]
    val_data = images[train_count:train_count + val_count]
    test_data = images[train_count + val_count:]

    for dataset, dataset_name in zip([train_data, val_data, test_data], ['train', 'val', 'test']):
        if not os.path.exists(os.path.join(image_dir, dataset_name)):
            os.mkdir(os.path.join(image_dir, dataset_name))
       
        for image_file in dataset:
            shutil.move(os.path.join(image_dir, image_file), os.path.join(image_dir, dataset_name, image_file))
   


def create_train_csv_triangle_multiple_times(dataset_directory, csv_dir):
    # dataset_directory: the training dir
    # num_category: how many categories we have in general (not sub category)
    # sample multiple times training set for a large training set
    for n in range(10):
       
        data = {'filepath': [], 'para0': [], 'para1': [], 'para2': []}    
        csv_filename = os.path.join(csv_dir, "train_%s.csv"%(n))  # train.csv test.csv val.csv
        print(csv_filename)
        # sample with replacement

        jpg_files = glob.glob(dataset_directory + '/*.jpg')

        num_jpg_files = len(jpg_files)
        sample_num_jpg_files = int(num_jpg_files*0.7)

 
        for file in random.sample(glob.glob(dataset_directory + '/*.jpg'), sample_num_jpg_files):
            filename = file.split('\\')[-1]
            values = filename.split("_")
            num_labels = len(values) #id and its value are the first two
            data['filepath'].append(file)
            for i in range(2, num_labels):
                if i == num_labels-1:
                    value = int(values[i].split(".")[0])
                else: 
                    value = int(values[i])
                key = 'para' + str(i-2)   
                print(key)              
                data[key].append(value)
        

        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(csv_filename, index=False)


def create_test_val_csv_triangle_multiple_times(test_directory, val_directory, csv_dir):
    # dataset_directory: the testing dir, the val dir
    # sample multiple times training set for a large training set
    for dataset_directory in [test_directory,val_directory]:
    
    
        data = {'filepath': [], 'para0': [], 'para1': [], 'para2': []}    
       
        if dataset_directory == test_directory:
            csv_filename = os.path.join(csv_dir, "test.csv")  # test.csv val.csv
        else:
            csv_filename = os.path.join(csv_dir, "val.csv")
        print(csv_filename)
        # sample with replacement
        jpg_files = glob.glob(dataset_directory + '/*.jpg')
        for file in jpg_files:

            filename = file.split('\\')[-1]
            values = filename.split("_")

            num_labels = len(values) #id and its value are the first two
            data['filepath'].append(file)
            for i in range(2, num_labels):
                if i == num_labels-1:
                    value = int(values[i].split(".")[0])
                else: 
                    value = int(values[i])
                key = 'para' + str(i-2)   
                print(key)              
                data[key].append(value)

        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(csv_filename, index=False)




def create_csv_realdata(dataset_directory, num_category):
   
    # this is for generate the csv from the real data from e.g., dear data
  
 
    data = {'filepath': [], 'para0': [], 'para1': [], 'para2': []}    
  
    csv_filename = os.path.join(dataset_directory,   "test.csv")  # train.csv test.csv val.csv
    print(csv_filename)
    files = os.listdir(dataset_directory)
    for file in files:
        if file.lower().endswith(('_test.jpg')):  # Check for image files
            path = os.path.join(dataset_directory, file)
            #value0 = value1 = value2 = value3 = 0
            data['filepath'].append(path)
            for i in range(num_category): 
                value = 0
                key = 'para' + str(i)                    
                data[key].append(value)

    # Create a DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    df.to_csv(csv_filename, index=False)


design_name = 'thoughts'
dataset_dir = ".\\thoughts_rec_dataset"
realdata_dir  = "..\glyph_detection\\thoughts_detection_results"
num_category=3






split_data(dataset_dir, train_split=0.7, val_split=0.1, test_split=0.2)
train_directory = os.path.join(dataset_dir, "train")   # Change this to your images' folder path



csv_dir = dataset_dir

create_train_csv_triangle_multiple_times(train_directory, csv_dir)

create_csv_realdata(realdata_dir, num_category)

test_directory =  os.path.join(dataset_dir, "test")
val_directory=  os.path.join(dataset_dir, "val")
create_test_val_csv_triangle_multiple_times(test_directory, val_directory, csv_dir)