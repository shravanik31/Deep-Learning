Video caption generation
The goal for this assignment is to develop a model that can automatically generate descriptive captions for short videos, considering the diverse attributes of the videos, such as varying objects and actions, and handling the variable length of both the input videos and the output captions.


Execution Process:
Training:
Initially, training is performed on the model by running the below command:

python3 training.py /scratch/shravak/HW2/MLDS_hw2_1_data/training_data/feat /scratch/shravak/HW2/MLDS_hw2_1_data/training_label.json

I have specified my respective paths to the training data features folder and training_label.json file in the command. If you want to perform the training, please specify your paths to the respective folders in the same order as in the command.

A trained model with name “model_shravani.h5” is saved at the end of training process.

Testing:
Before starting the testing process, please download the pretrained model “model_shravani.h5”, testing_label.json, index_to_word.pickle and blue_eval.py files to your respective directory.

To test the model, the following shell script (hw2_seq2seq.sh) is run with the below command:

sh hw2_seq2seq.sh /scratch/shravak/HW2/MLDS_hw2_1_data/testing_data output_captions.txt

I have specified my respective paths to the testing data folder and output_captions.txt file in the command. If you want to perform the testing, please specify your paths to the respective folders in the same order as in the command.

After the testing process is completed, the resulting captions are stored into an output file.

Results:
The average BLEU score that I have obtained is 0.625.
