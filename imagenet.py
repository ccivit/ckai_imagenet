from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import model_from_yaml
import numpy as np
import sys
import os
import yaml


def save_results(preds,img_name,img_path,config):
    pipeline_dir = config['general']['pipeline_path']
    results_filename = config['imagenet']['results']
    cleaned_results = {}
    for item in decode_predictions(preds,top=5)[0]:
        cleaned_results[str(item[1])] = float(item[2])
    print(cleaned_results)
    img_name = remove_file_extension(img_name)
    img_dir_wo_parent = get_path_wo_top_parent_dirs(img_path)
    pipeline_path = os.path.join(pipeline_dir,img_dir_wo_parent,img_name)
    results_filepath = os.path.join(pipeline_path,results_filename)
    print(results_filepath)
    print('********************************')
    yaml.dump(cleaned_results, open(results_filepath, "w"), default_flow_style=False)
    return None

def remove_file_extension(file_name):
    return '.'.join(file_name.split('.')[:-1])

def get_path_wo_top_parent_dirs(file_path):
    return '/'.join(os.path.split(file_path)[0].split('/')[-3:])

def load_pretrained_model(model_filename):
    try:
        yaml_file = open(model_filename, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        model.load_weights(model_h5)
        print("Loaded model from disk")

    except:
        print("Couldn't find models locally")
        model = ResNet50(weights='imagenet')
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open(model_filename, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights(model_h5)
        print("Saved model to disk")
    return model


img_dir = sys.argv[1]
config_file = sys.argv[2]

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
    model_filename = config['imagenet']['model']
    model_h5 = config['imagenet']['weights']
    # results_filename = config['imagenet']['results']
    # pipeline_dir = config['general']['pipeline_path']

model = load_pretrained_model(model_filename)

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img_name)
    print(img_path)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except OSError:
        continue

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    save_results(preds,img_name,img_path,config)
