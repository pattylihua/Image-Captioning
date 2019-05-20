To run the training:
1.Run “resize.py” to resize the each image. 
Store the images at “./data/resized_train2014” and “./data/resized_val2014” folders. 
2.Run “image_captioning.py”.
Store the best model after epochs in “./data/models/” with name “best_model_encoder.pth” and “best_model_decoder.pth”.
3.To adjust parameters. 
Open “image_captioning.py” file, adjust the parameters under ‘if __name__ == '__main__' . 
* Node:
1.Please ensure the data is in current directory under “/data” folder, i.e “./data/*”
2.“/train2014” folder contains all train images, “/val2014” folder contains all validation images and “/annotations” folders contains “captions_train2014.json” and ‘captions_val2014.json” json files in order to load the captions.

To run the GUI:
1. Run "gui.py". (there is a testing encoder and decoder model and vocab.pkl in the folder, if you want just run the GUI with any changes, please go ahead.)
2. Select a image.
3. A window with the image and predicted captions will show at top of the image.
* Please ensure there is an encoder model 'pth' file and an decoder model 'pth' file as well as the vocab.pkl file in your current directory to run the GUI successfully.