# Basic-Image-Classifier-Website

I have used pretrained YOLO-v2 algorith for Object detection and classification

Follow following steps to use the above code:

* open command prompt in desired directory and executes following commands
```
Install Darkflow
$ pip install Cython
$ git clone https://github.com/thtrieu/darkflow.git
$ cd darkflow
$ python3 setup.py build_ext --inplace
$ pip install .
```
* Download weights from [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU)
* Prepare a bin folder in Darkflow and download above weights(yolo.weights) in the that bin folder.
* clone above repository in darkflow folder.
* Open command prompt and run '$ python yolomodel.py' and boom you have your own web page for Image Classification.
* Just make sure you have imported all the python libraries specified in Requirement.txt
