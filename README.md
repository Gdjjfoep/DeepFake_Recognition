1.RUN THIS ONE
pip install tensorflow[and-cuda] opencv-python scikit-learn numpy

2.FOLDER STRUCTURE
/Project_Folder
│
├── run_training.py      <-- Run this file!
├── train_model.py       <-- Architecture (EfficientNet + LSTM)
├── video_utils.py       <-- Video processing helpers
│
└── dataset/             <-- 18GB Dataset Folder
    ├── real/            <-- Contains real .mp4 videos
    └── fake/            <-- Contains deepfake .mp4 videos



3. FOR DATASET, WILL BE TRANSFERED VIA TOFFEESHARE(18GB CHA AHE)
   IT HAS 1000S OF VIDEOS WITH NO AUDIO IN EACH FOLDER
   JUST USE THE VIDEOS FRM ORIGNAL FOR REAL
   & USE THE DEEPFAKE FOR FAKE 


4. ONCE DONE WITH DATASET DOWNLOADING, MAKE SURE THAT BOTH THE, REAL-FAKE FOLDERS HAVE RESPECTIVE VIDEOS IN IT

5. AFTER DOING ALL OF THIS SHIT, RUN THE  "run_training.py" FILE, N LET IT EXECTUTE IT UNTIL THE TRANING IS DONE

6. ONCE TARINING IS DONE CALL ME 
